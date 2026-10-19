"""Partial Conv2d/Linear arithmetic count, unchanged from the full release."""
import torch

def count(model):
    macs=0;handles=[]
    def hook(m,args,out):
        nonlocal macs
        if isinstance(m,torch.nn.Conv2d):macs+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,torch.nn.Linear):macs+=out.numel()*m.in_features
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():model(torch.zeros(1,3,32,32))
    for h in handles:h.remove()
    return dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),conv_linear_MACs=macs,
                conv_linear_arithmetic_FLOPs=2*macs,total_model_FLOPs=None)
