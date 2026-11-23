"""Minibatch training, validation selection and resumable checkpoints."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import EpochSampler, FlamePairs, PairedScaleShift
from .io import save_checkpoint, write_csv, write_json
from .losses import segmentation_loss
from .metrics import NAMES, color_mask, confusion, metrics_from_confusion
from .model import CRAGSeg

def setup(seed,c,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(c["cpu_threads"])
    if device.type=="cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True

def autocast(c,device):
    return torch.autocast(device.type,dtype=getattr(torch,c["precision"]),
                          enabled=c["amp"] and device.type=="cuda")

def checkpoint_config(c,seed):
    return dict(config=c,model="CRAGSeg",seed=seed)

def dataset(data_dir,c,split,seed):
    list_file = data_dir / c["split_dir"] / (split + "_flm.txt")
    pairs = FlamePairs(data_dir, list_file, c["image_size"], split == "train", seed)
    return PairedScaleShift(pairs,seed) if split == "train" else pairs

def make_loader(data,c,device,seed=200,training=False):
    sampler=EpochSampler(data,seed) if training else None
    workers=c["workers"] if training else 0
    options=dict(num_workers=workers,pin_memory=device.type=="cuda",
                 generator=torch.Generator().manual_seed(seed),
                 batch_size=c["micro_batch"] if training else c["eval_batch"],
                 drop_last=training,sampler=sampler)
    if workers:options.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(data,**options),sampler

def state_rng(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type=="cuda" else None)

def convolution_channels_last(model):
    """Optimize 2-D convolution weights without converting arbitrary buffers.

    EvidenceInteraction.evidence_masks has shape (1, 2, 8, 1, 1). A whole-model
    .to(memory_format=torch.channels_last) tries to convert that 5-D buffer to a
    4-D-only format. The selective utility preserves masks and graph tensors.
    Parameter values, checkpoint keys and the model's computation are unchanged.
    """
    return torch.nn.utils.convert_conv2d_weight_memory_format(model,torch.channels_last)

def restore_rng(state,device):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if device.type=="cuda" and state["cuda"] is not None:torch.cuda.set_rng_state_all(state["cuda"])

def best_record(model,config,epoch,score):
    return dict(model=model.state_dict(),run= config,best_epoch=epoch,best_val_miou=score)

def validate(model,loader,c,device,destination=None,observe_mask=None):
    model.eval();cm=np.zeros((3,3),dtype=np.int64);loss_sum=0.;count=0;rows=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for images,labels,names in loader:
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
            with autocast(c,device):
                out=model(images);loss,_=segmentation_loss(out,labels)
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite validation loss")
            predictions=out["logits"].argmax(1)
            cm+=confusion(predictions,labels).cpu().numpy()
            count+=len(names);loss_sum+=float(loss)*len(names)
            if destination is not None:
                for i,name in enumerate(names):
                    individual=confusion(predictions[i],labels[i]).cpu().numpy()
                    rows.append(dict(sample=name,**metrics_from_confusion(individual)))
                    if observe_mask is not None:
                        observe_mask(name,predictions[i].cpu().numpy(),labels[i].cpu().numpy())
                    if destination.name=="test":
                        mask=predictions[i].cpu().numpy().astype(np.uint8)
                        true=labels[i].cpu().numpy()
                        color=color_mask(mask)
                        gt=color_mask(np.minimum(true,2));gt[true==255]=[255,0,255]
                        im=(images[i].float().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
                        rgb,ir=im[:,:,:3],np.repeat(im[:,:,3:4],3,2)
                        for folder in ("ids","color","panels"):(destination/"predictions"/folder).mkdir(parents=True,exist_ok=True)
                        fname=Path(name).stem.replace("XXX","pred")+".png"
                        Image.fromarray(mask).save(destination/"predictions/ids"/fname)
                        Image.fromarray(color).save(destination/"predictions/color"/fname)
                        Image.fromarray(np.concatenate([rgb,ir,gt,color],1)).save(destination/"predictions/panels"/fname)
    result=dict(**metrics_from_confusion(cm),loss=loss_sum/max(count,1),
                samples=count,wall_seconds=time.perf_counter()-started)
    if destination is not None:
        write_csv(destination/"per_image.csv",rows)
        write_csv(destination/"confusion.csv",[dict(true_class=NAMES[i],**{NAMES[j]:int(cm[i,j]) for j in range(3)}) for i in range(3)])
    return result

def learning_rate(c,epoch,update,updates):
    progress=epoch+update/updates
    if progress<c["warmup_epochs"]:
        return c["lr"]*(epoch*updates+update+1)/(c["warmup_epochs"]*updates)
    ratio=(progress-c["warmup_epochs"])/max(c["epochs"]-c["warmup_epochs"],1)
    factor=c["min_lr_ratio"]+(1-c["min_lr_ratio"])*.5*(1+math.cos(math.pi*min(1.,ratio)))
    return c["lr"]*factor

def train_one(data_dir,out,c,seed,device,resume):
    folder=out/f"seed_{seed}";folder.mkdir(parents=True,exist_ok=True)
    run=checkpoint_config(c,seed)
    run["split_sha256"] = {split: hashlib.sha256((data_dir / c["split_dir"] / (split+"_flm.txt")).read_bytes()).hexdigest() for split in ("train","val")}
    completed=folder/"train_complete.json";latest=folder/"weights/latest.pth";best_path=folder/"weights/best.pth"
    if completed.exists():
        done=json.loads(completed.read_text(encoding="utf-8"))
        if done["run"]!=run or not best_path.exists():raise RuntimeError("Completed run configuration/weights mismatch: "+str(folder))
        print("Verified completed training:",folder.name,flush=True);return
    if latest.exists() and not resume:raise RuntimeError("Existing checkpoint. Use --resume.")
    setup(seed,c,device)
    model=CRAGSeg().to(device)
    if device.type=="cuda":model=convolution_channels_last(model)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=c["lr"],weight_decay=c["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and c["amp"] and c["precision"]=="float16")
    history=[];start=0;best=-math.inf;best_epoch=-1;stale=0
    if latest.exists():
        state=torch.load(latest,map_location="cpu",weights_only=False)
        if state["run"]!=run:raise RuntimeError("Resume config differs; use a new output directory.")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scaler.load_state_dict(state["scaler"])
        history=state["history"];start=state["epoch"]+1;best=state["best_val_miou"];best_epoch=state["best_epoch"];stale=state["stale"]
        # latest is committed before best; recovery completes an interrupted best save.
        saved_best=torch.load(best_path,map_location="cpu",weights_only=False) if best_path.exists() else None
        if saved_best is None or saved_best["best_epoch"]!=best_epoch:
            if best_epoch!=state["epoch"]:raise RuntimeError("Best checkpoint is missing or inconsistent.")
            save_checkpoint(best_path,best_record(model,run,best_epoch,best))
        restore_rng(state["rng"],device)
        del state,saved_best
    train_loader,sampler=make_loader(dataset(data_dir,c,"train",seed),c,device,seed,True)
    val_loader,_=make_loader(dataset(data_dir,c,"val",seed),c,device,seed)
    accumulation=c["effective_batch"]//c["micro_batch"]
    batches=(len(train_loader)//accumulation)*accumulation
    if not batches:raise ValueError("Insufficient training samples for one effective batch.")
    updates=batches//accumulation
    if history:write_csv(folder/"history.csv",history)
    write_json(folder/"run_config.json",run)
    for epoch in range(start,c["epochs"]):
        if epoch>=c["min_epochs"] and stale>=c["patience"]:break
        model.train();model.set_epoch(epoch);sampler.epoch=epoch
        train_loss=0.;parts={};t0=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        for index,(images,labels,_) in enumerate(train_loader):
            if index>=batches:break
            lr=learning_rate(c,epoch,index//accumulation,updates)
            for group in optimizer.param_groups:group["lr"]=lr
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
            with autocast(c,device):
                predictions=model(images);loss,components=segmentation_loss(predictions,labels)
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite training loss; checkpoint preserved.")
            scaler.scale(loss/accumulation).backward()
            if (index+1)%accumulation==0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            train_loss+=float(loss.detach())
            for key,value in components.items():parts[key]=parts.get(key,0.)+float(value)
            if (index+1)%20==0 or index+1==batches:
                print(f"{folder.name} epoch {epoch+1}/{c['epochs']} batch {index+1}/{batches} loss={train_loss/(index+1):.4f}",flush=True)
        validation=validate(model,val_loader,c,device)
        score=validation["miou"]
        if not math.isfinite(score):raise FloatingPointError("Invalid validation mIoU")
        improved=score>best+c["min_delta"]
        if improved:best,best_epoch,stale=score,epoch,0
        else:stale+=1
        row=dict(epoch=epoch+1,train_loss=train_loss/batches,**{"train_"+k:v/batches for k,v in parts.items()},
                 **{"val_"+k:v for k,v in validation.items()},lr=lr,best_val_miou=best,
                 best_epoch=best_epoch+1,stale=stale,seconds=time.perf_counter()-t0,optimizer_updates=updates)
        history.append(row)
        record=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    run=run,epoch=epoch,best_epoch=best_epoch,best_val_miou=best,stale=stale,history=history,rng=state_rng(device))
        save_checkpoint(latest,record)
        if improved:save_checkpoint(best_path,best_record(model,run,best_epoch,best))
        write_csv(folder/"history.csv",history)
        write_json(folder/"progress.json",dict(phase="train",run=folder.name,epoch=epoch+1,max_epochs=c["epochs"],
                   best_epoch=best_epoch+1,best_val_miou=best,stale=stale,patience=c["patience"],updated=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"Validation: {folder.name} epoch={epoch+1} mIoU={score:.6f} best={best:.6f} stale={stale}/{c['patience']}",flush=True)
    if not best_path.exists():raise RuntimeError("Training produced no best checkpoint")
    write_json(completed,dict(run=run,best_epoch=best_epoch+1,best_val_miou=best,
                              epochs_completed=len(history),reason="early_stop" if len(history)<c["epochs"] else "max_epochs"))
    del train_loader,val_loader,model,optimizer,scaler
    if device.type=="cuda":torch.cuda.empty_cache()
