"""Paired FLAME2 images, palette decoding and deterministic epoch sampling."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
cv2.setNumThreads(0)

def read_list(path):
    lines=[s.strip().replace("\\","/") for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if len(lines)!=len(set(lines)): raise ValueError("Duplicated samples: "+str(path))
    return lines

def decode_mask(raw):
    out=np.full(raw.shape[:2],255,dtype=np.uint8)
    for i,c in enumerate((0,125,255)): out[np.all(raw==c,axis=-1)]=i
    return out

class FlamePairs(Dataset):
    def __init__(self,data_dir,list_file,size=256,training=False,seed=200,limit=None):
        self.root=Path(data_dir); self.samples=read_list(list_file)
        if limit is not None:
            indices=np.linspace(0,len(self.samples)-1,min(limit,len(self.samples)),dtype=int)
            self.samples=[self.samples[i] for i in indices]
        self.size=size; self.training=training; self.seed=seed
    def __len__(self):return len(self.samples)
    def __getitem__(self,key):
        index,epoch=key if isinstance(key,tuple) else (key,0)
        name=self.samples[index]
        with Image.open(self.root/name.replace("XXX","rgb")) as im:r=np.asarray(im.convert("RGB")).copy()
        with Image.open(self.root/name.replace("XXX","ir")) as im:t=np.asarray(im.convert("L")).copy()
        with Image.open(self.root/name.replace("XXX","gt")) as im:y=decode_mask(np.asarray(im.convert("RGB")))
        if r.shape[:2]!=t.shape or t.shape!=y.shape:raise ValueError("Unaligned triplet: "+name)
        s=self.size
        if self.training:
            h,w=y.shape
            if h>w: nh,nw=s,int(w*s/h+.5)
            else: nh,nw=int(h*s/w+.5),s
            r=cv2.resize(r,(nw,nh));t=cv2.resize(t,(nw,nh));y=cv2.resize(y,(nw,nh),interpolation=cv2.INTER_NEAREST)
            dh,dw=max(0,s-nh),max(0,s-nw)
            r=cv2.copyMakeBorder(r,0,dh,0,dw,cv2.BORDER_CONSTANT,value=0)
            t=cv2.copyMakeBorder(t,0,dh,0,dw,cv2.BORDER_CONSTANT,value=0)
            y=cv2.copyMakeBorder(y,0,dh,0,dw,cv2.BORDER_CONSTANT,value=255)
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,index]))
            oy=int(rng.integers(0,y.shape[0]-s+1));ox=int(rng.integers(0,y.shape[1]-s+1))
            r=r[oy:oy+s,ox:ox+s];t=t[oy:oy+s,ox:ox+s];y=y[oy:oy+s,ox:ox+s]
            if rng.integers(0,2):r=r[:,::-1];t=t[:,::-1];y=y[:,::-1]
        else:
            r=cv2.resize(r,(s,s));t=cv2.resize(t,(s,s));y=cv2.resize(y,(s,s),interpolation=cv2.INTER_NEAREST)
        x=np.concatenate([r,t[:,:,None]],axis=2).astype(np.float32)/255.
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y.copy()).long(),name

class EpochSampler(Sampler):
    def __init__(self,dataset,seed):self.dataset=dataset;self.seed=seed;self.epoch=0
    def __iter__(self):
        order=torch.randperm(len(self.dataset),generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        return iter((i,self.epoch) for i in order)
    def __len__(self):return len(self.dataset)

def audit_data(root,split_dir,expected):
    root,split_dir=Path(root),Path(split_dir)
    seen=set();records={};split_hashes={}
    for split in ("train","val","test","buffer_excluded"):
        f=split_dir/(split+"_flm.txt")
        names=read_list(f)
        if seen.intersection(names):raise ValueError("Split overlap: "+split)
        seen.update(names)
        split_hashes[split]=hashlib.sha256(f.read_bytes()).hexdigest()
        if split=="buffer_excluded":continue
        if len(names)!=expected[split]:raise ValueError("Unexpected split count: "+split)
        invalid=0
        for name in names:
            if ".." in Path(name).parts or Path(name).is_absolute() or name.count("XXX")!=1:
                raise ValueError("Invalid data-list entry: "+name)
            shapes=[]
            for mode in ("rgb","ir","gt"):
                with Image.open(root/name.replace("XXX",mode)) as im:
                    im.load();shapes.append(im.size)
                    if mode=="gt":
                        mask=decode_mask(np.asarray(im.convert("RGB")))
                        invalid+=int((mask==255).sum())
                        if np.all(mask==255):raise ValueError("Mask has no valid pixels: "+name)
            if len(set(shapes))!=1:raise ValueError("Dimension mismatch: "+name)
        records[split]={"samples":len(names),"unknown_palette_pixels":invalid}
    return {"splits":records,"list_sha256":split_hashes,"classes":{"0":"background","1":"smoke","2":"fire"},
            "evaluation":"whole frame resized to the configured image size",
            "source":"CRAGSeg grouped FLAME2 split"}


class PairedScaleShift(Dataset):
    """Apply a shared training-only scale and shift to each triplet with probability 0.5."""
    def __init__(self, base, seed):
        self.base = base
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        images, labels, name = self.base[key]
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        # A separate RNG stream leaves the base sampler and its flip unchanged.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index, 40805]))
        if rng.random() >= 0.5:
            return images, labels, name

        h, w = labels.shape
        scale = rng.uniform(0.95, 1.10)
        shift_x = int(round(0.04 * w))
        shift_y = int(round(0.04 * h))
        transform = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0.0, scale)
        transform[0, 2] += int(rng.integers(-shift_x, shift_x + 1))
        transform[1, 2] += int(rng.integers(-shift_y, shift_y + 1))

        image_array = np.ascontiguousarray(images.permute(1, 2, 0).numpy())
        augmented = cv2.warpAffine(image_array, transform, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(labels.numpy().astype(np.uint8), transform, (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)
        return torch.from_numpy(augmented).permute(2, 0, 1).contiguous(), torch.from_numpy(mask).long(), name
