import csv,json,math,os
from pathlib import Path
import numpy as np
import torch

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not math.isfinite(v):return None
    if isinstance(v,np.generic):return v.item()
    return v

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False),encoding="utf-8")
    os.replace(tmp,path)

def write_csv(path,rows,fieldnames=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows and not fieldnames:raise ValueError("Cannot write a schema-less empty CSV")
    keys=fieldnames or list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow(clean(row))
    os.replace(tmp,path)

def save_checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp");torch.save(state,tmp);os.replace(tmp,path)
