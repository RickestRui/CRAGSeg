import torch
from torch.nn import functional as F

def boundary_target(labels):
    valid = labels != 255
    edge = torch.zeros_like(labels,dtype=torch.bool)
    h = (labels[:,:,1:]!=labels[:,:,:-1]) & valid[:,:,1:] & valid[:,:,:-1]
    v = (labels[:,1:,:]!=labels[:,:-1,:]) & valid[:,1:,:] & valid[:,:-1,:]
    edge[:,:,1:] |= h; edge[:,:,:-1] |= h
    edge[:,1:,:] |= v; edge[:,:-1,:] |= v
    edge = F.max_pool2d(edge[:,None].float(),3,1,1)[:,0]
    return edge,valid

def auxiliary_loss(logits,labels):
    valid=(labels!=255)[:,None].float()
    one=F.one_hot(labels.clamp(0,2),3).permute(0,3,1,2).float()*valid
    pooled=F.adaptive_avg_pool2d(one,logits.shape[-2:])
    mass=F.adaptive_avg_pool2d(valid,logits.shape[-2:])
    target=pooled/mass.clamp_min(1e-6)
    loss=-(target*logits.float().log_softmax(1)).sum(1,True)
    return (loss*mass).sum()/mass.sum().clamp_min(1)

def segmentation_loss(out,labels):
    logits=out["logits"].float()
    valid=labels!=255
    ce=F.cross_entropy(logits,labels,ignore_index=255,reduction="none")
    ce=(ce*valid).sum()/valid.sum().clamp_min(1)
    p=logits.softmax(1)*valid[:,None]
    one=F.one_hot(labels.clamp(0,2),3).permute(0,3,1,2)*valid[:,None]
    axes=(0,2,3)
    dice=1-((2*(p*one).sum(axes)+1)/(p.sum(axes)+one.sum(axes)+1)).mean()
    aux=.1*(auxiliary_loss(out["rgb_aux"],labels)+auxiliary_loss(out["ir_aux"],labels))
    edge,mask=boundary_target(labels)
    b=out["boundary"][:,0].float()
    pos=((mask.sum()-edge[mask].sum())/edge[mask].sum().clamp_min(1)).clamp(1,20).detach()
    bl=F.binary_cross_entropy_with_logits(b,edge,pos_weight=pos,reduction="none")
    bl=(bl*mask).sum()/mask.sum().clamp_min(1)
    total=ce+.5*dice+aux+.1*bl
    return total,{"ce":ce.detach(),"dice":dice.detach(),"aux":aux.detach(),"boundary":bl.detach()}
