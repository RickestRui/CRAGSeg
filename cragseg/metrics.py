import numpy as np
import torch
NAMES=("background","smoke","fire")

def confusion(pred,target):
    keep=(target>=0)&(target<3)
    return torch.bincount((target[keep]*3+pred[keep]).reshape(-1),minlength=9).reshape(3,3)

def metrics_from_confusion(matrix):
    c=np.asarray(matrix,dtype=np.float64)
    tp=c.diagonal();actual=c.sum(1);pred=c.sum(0);union=actual+pred-tp
    divide=lambda a,b:np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
    iou=divide(tp,union);f1=divide(2*tp,actual+pred)
    precision=divide(tp,pred);recall=divide(tp,actual)
    result={"macro_recall":float(np.nanmean(recall)), "miou":float(np.nanmean(iou)),"macro_f1":float(np.nanmean(f1)),
            "pixel_accuracy":float(tp.sum()/c.sum()) if c.sum() else float("nan"),
            "foreground_miou":float(np.nanmean(iou[1:])) if np.any(union[1:]>0) else float("nan")}
    for i,name in enumerate(NAMES):
        result.update({name+"_iou":float(iou[i]),name+"_f1":float(f1[i]),
                       name+"_precision":float(precision[i]),name+"_recall":float(recall[i])})
    return result

def color_mask(mask):
    return np.asarray([[0,0,0],[157,164,176],[255,109,31]],dtype=np.uint8)[mask]
