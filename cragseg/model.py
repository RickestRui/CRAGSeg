"""Cross-modal region aggregation for fire and smoke segmentation."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .graph import RegionGraphBuilder, GraphBlock

def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def gradient(x):
    dx = F.pad((x[..., 1:] - x[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy).clamp(0, 1)

def measured_evidence(images):
    rgb, ir = images[:, :3], images[:, 3:4]
    red = rgb[:, :1] - .5 * (rgb[:, 1:2] + rgb[:, 2:3])
    contrast = ir - F.avg_pool2d(F.pad(ir, (4, 4, 4, 4), mode="replicate"), 9, 1)
    return torch.cat([rgb, ir, red, contrast, gradient(rgb.mean(1, True)), gradient(ir)], 1)

class ConvNorm(nn.Sequential):
    def __init__(self, ci, co, kernel=3, groups=1):
        super().__init__(nn.Conv2d(ci, co, kernel, padding=kernel//2, groups=groups, bias=False),
                         nn.GroupNorm(8, co), nn.SiLU())

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        net = resnet18(weights=None)
        if channels == 1:
            net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.s4, self.s8, self.s16, self.s32 = net.layer1, net.layer2, net.layer3, net.layer4
    def early(self, x, return_half=False):
        # Preserve the original stem order. Retain pre-pool features only on request.
        half = self.stem[2](self.stem[1](self.stem[0](x)))
        a = self.s4(self.stem[3](half)); b = self.s8(a); c = self.s16(b)
        return (half, a, b, c) if return_half else (a, b, c)


def reciprocal_edge_mask(source, target, node_count):
    """Select a directed sparse edge only when the reverse edge was also proposed."""
    if source.numel() == 0:
        return torch.empty_like(source, dtype=torch.bool)
    keys = (source * node_count + target).sort().values
    reverse = target * node_count + source
    positions = torch.searchsorted(keys, reverse).clamp_max(keys.numel() - 1)
    return keys[positions] == reverse

class EvidenceInteraction(nn.Module):
    def __init__(self, hidden=128, local_k=4, cross_k=2):
        super().__init__()
        self.project = nn.Conv2d(256, hidden, 1)
        self.auxiliary = nn.Conv2d(hidden, 3, 1)
        self.embed = nn.Sequential(nn.Linear(hidden + 16 + 3 + 1 + 1 + 2 + 2, hidden),
                                   nn.LayerNorm(hidden), nn.SiLU())
        self.builder = RegionGraphBuilder(local_k, 256)
        self.cross_k = cross_k
        self.layers = nn.ModuleList([GraphBlock(hidden, 4, 16, .1) for _ in range(2)])
        self.back = nn.Conv2d(hidden, 256, 1)
        nn.init.normal_(self.back.weight, std=.001)
        nn.init.zeros_(self.back.bias)
        self.gate = nn.Conv2d(hidden*2 + 1, 1, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.)
        self.register_buffer("evidence_masks", torch.tensor([[1,1,1,0,1,0,1,0],
                                                            [0,0,0,1,0,1,0,1.]]).view(1,2,8,1,1))

    def forward(self, rgb, ir, images, strength=1., details=False):
        b, _, gh, gw = rgb.shape
        n = gh * gw
        projected = torch.stack([self.project(rgb), self.project(ir)], 1)
        flat_projected = projected.flatten(0, 1)
        aux = self.auxiliary(flat_projected).reshape(b, 2, 3, gh, gw)
        evidence = measured_evidence(images)[:, None] * self.evidence_masks
        evidence = evidence.flatten(0, 1)
        def pool(t, maximum=False):
            fn = F.adaptive_max_pool2d if maximum else F.adaptive_avg_pool2d
            return fn(t, (gh, gw)).flatten(2).transpose(1, 2)
        ev = pool(evidence)
        p = aux.detach().flatten(0, 1).softmax(1).flatten(2).transpose(1, 2)
        en = -(p*p.clamp_min(1e-7).log()).sum(-1, True)/math.log(3)
        boundary = pool(gradient(images[:,:3].mean(1,True)), True)
        ir_boundary = pool(gradient(images[:,3:4]), True)
        boundary = torch.stack([boundary, ir_boundary], 1).reshape(2*b,n,1)
        feat = flat_projected.flatten(2).transpose(1, 2)
        desc = torch.cat([F.normalize(feat,dim=-1), 2*p], -1)
        # Existing local/self/nonlocal implementation isolates each image and modality.
        edges, attrs, coords = self.builder(desc, p, boundary, (gh,gw))
        types = attrs[:,-3:].argmax(-1)
        # Cross-modal links are directed: opposite modality -> current target.
        cross_s, cross_d, cross_t = [], [], []
        with torch.no_grad():
            d = F.normalize(desc.detach(),dim=-1).reshape(b,2,n,-1)
            ids = torch.arange(n,device=rgb.device)
            off = torch.arange(b,device=rgb.device)[:,None]*2*n
            for target_m in (0,1):
                target = off + target_m*n + ids
                source = off + (1-target_m)*n + ids
                cross_s.append(source.flatten()); cross_d.append(target.flatten())
                cross_t.append(torch.full_like(source.flatten(),3))
                k = min(self.cross_k, max(n-1,0))
                if k:
                    score = torch.bmm(d[:,target_m], d[:,1-target_m].transpose(1,2))
                    score[:,ids,ids] = float("-inf")
                    near = score.topk(k,dim=-1).indices
                    cross_s.append((near+off[:,:,None]+(1-target_m)*n).flatten())
                    cross_d.append(target[:,:,None].expand(-1,-1,k).flatten())
                    cross_t.append(torch.full_like(cross_s[-1],4))
        src = torch.cat([edges[0],*cross_s]); dst = torch.cat([edges[1],*cross_d])
        types = torch.cat([types,*cross_t])
        candidates = types == 4
        keep = ~candidates
        keep[candidates] = reciprocal_edge_mask(src[candidates], dst[candidates], b * 2 * n)
        src, dst, types = src[keep], dst[keep], types[keep]
        edge_index = torch.stack([src,dst])
        pos = coords.repeat(b*2,1)
        pos_delta = pos[src]-pos[dst]
        prob = p.flatten(0,1); ent = en.flatten(0,1)
        raw = ev.flatten(0,1); bd = boundary.flatten(0,1)
        norm = F.normalize(desc,dim=-1).flatten(0,1)
        pdiff = (prob[src]-prob[dst]).abs()
        # Do not compare RGB color directly with IR intensity across modalities.
        ediff = (raw[src]-raw[dst]).abs().mean(-1,True)*(types<3)[:,None]
        barrier = torch.maximum(bd[src],bd[dst])*(types!=2)[:,None]
        edge_attr = torch.cat([pos_delta,pos_delta.norm(dim=-1,keepdim=True),
                               (norm[src]*norm[dst]).sum(-1,True),pdiff,
                               ent[src],ent[dst],barrier,ediff,
                               F.one_hot(types,5).to(rgb.dtype)],-1)
        # A soft reliability bias; uncertain or conflicting links remain possible.
        prior = -.5*ent[src,0] - .5*pdiff.mean(-1) - .25*ediff[:,0]
        prior = torch.where(types==2,torch.zeros_like(prior),prior)
        modality = F.one_hot(torch.arange(2,device=rgb.device),2).to(rgb.dtype)
        modality = modality[None,:,None].expand(b,2,n,2).reshape(2*b,n,2)
        z0 = self.embed(torch.cat([feat,ev,pool(evidence,True),p,en,boundary,
                                   coords[None].expand(2*b,-1,-1),modality],-1)).flatten(0,1)
        z = z0
        for layer in self.layers:
            z = layer(z,edge_index,edge_attr,prior)
        context = (z-z0).reshape(2*b,gh,gw,-1).permute(0,3,1,2).contiguous()
        gate = self.gate(torch.cat([flat_projected,context,
                                   en.transpose(1,2).reshape(2*b,1,gh,gw)],1)).sigmoid()
        delta = self.back(context)*gate*strength
        refined = torch.stack([rgb,ir],1)+delta.reshape(b,2,256,gh,gw)
        info = {"gate_mean":gate.detach().mean()}
        if details:
            info.update(edge_index=edge_index,edge_type=types,edge_attr=edge_attr,
                        nodes_per_modality=n,node_features=z,raw_evidence=ev)
        return refined[:,0],refined[:,1],aux[:,0],aux[:,1],info

class Fuse(nn.Module):
    def __init__(self, channels, width=128):
        super().__init__()
        self.rgb = ConvNorm(channels,width,1)
        self.ir = ConvNorm(channels,width,1)
        self.select = nn.Conv2d(width*2,width,1)
        self.mix = ConvNorm(width,width)
    def forward(self,rgb,ir):
        r,t = self.rgb(rgb),self.ir(ir)
        g = self.select(torch.cat([r,t],1)).sigmoid()
        return self.mix(g*r+(1-g)*t)

class HalfResolutionCorrection(nn.Module):
    """Predict a three-class residual from RGB/IR stem features and semantic logits."""

    def __init__(self, width=16):
        super().__init__()
        self.rgb = ConvNorm(64, width, kernel=1)
        self.ir = ConvNorm(64, width, kernel=1)
        self.mix = ConvNorm(2 * width + 3, width)
        self.output = nn.Conv2d(width, 3, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, rgb_half, ir_half, logits_quarter):
        logits_half = resize(logits_quarter, rgb_half.shape[-2:])
        inputs = torch.cat((self.rgb(rgb_half), self.ir(ir_half), logits_half), dim=1)
        return self.output(self.mix(inputs))


class CRAGSeg(nn.Module):
    """Segment background, smoke and fire from B x 4 x H x W RGB/IR tensors."""
    def __init__(self):
        super().__init__()
        self.rgb_encoder, self.ir_encoder = Encoder(3), Encoder(1)
        self.interaction = EvidenceInteraction()
        self.fuses = nn.ModuleList([Fuse(c) for c in (64,128,256,512)])
        self.decode = nn.ModuleList([ConvNorm(128,128) for _ in range(3)])
        self.detail = nn.Sequential(ConvNorm(128,128,3,128),ConvNorm(128,128,1))
        self.boundary_head = nn.Conv2d(128,1,1)
        self.classifier = nn.Conv2d(128,3,1)
        self.epoch = 0
        self.half_detail = HalfResolutionCorrection()

    def set_epoch(self,epoch):
        self.epoch = epoch
    def decode_quarter(self, images, rgb_features, ir_features, details):
        r4,r8,r16 = rgb_features
        t4,t8,t16 = ir_features
        strength = min(1.,max(0.,(self.epoch-4)/10.)) if self.training else 1.
        r16,t16,ar,at,info = self.interaction(r16,t16,images,strength,details)
        r32,t32 = self.rgb_encoder.s32(r16),self.ir_encoder.s32(t16)
        fused = [f(r,t) for f,r,t in zip(self.fuses,(r4,r8,r16,r32),(t4,t8,t16,t32))]
        x = fused[3]
        for block,skip in zip(self.decode,reversed(fused[:3])):
            x = block(resize(x,skip.shape[-2:])+skip)
        detail = self.detail(fused[0])
        boundary = self.boundary_head(detail)
        logits = self.classifier(x+detail)
        return logits, boundary, ar, at, info

    def forward(self, images, details=False):
        rgb_half, *rgb = self.rgb_encoder.early(images[:, :3], return_half=True)
        ir_half, *ir = self.ir_encoder.early(images[:, 3:4], return_half=True)
        coarse, boundary, rgb_aux, ir_aux, info = self.decode_quarter(images, rgb, ir, details)
        correction = self.half_detail(rgb_half, ir_half, coarse)
        size = images.shape[-2:]
        # Preserve the original quarter->full interpolation: quarter->half->full
        # would change the base prediction even with a zero correction.
        logits = resize(coarse, size) + resize(correction, size)
        return dict(logits=logits, rgb_aux=rgb_aux, ir_aux=ir_aux,
                    boundary=resize(boundary, size), **info)
