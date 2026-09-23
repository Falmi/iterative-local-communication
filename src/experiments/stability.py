"""Read-only observers for Phase 4; no gradients or model values are modified."""
import math
import torch
from src.experiments.metadata import write_json


class NumericalFailure(RuntimeError):
    pass


def scalar(x):
    v=float(x)
    return v if math.isfinite(v) else str(v)


class StabilityMonitor:
    def __init__(self, model, run, state=None):
        self.model,self.run=model,run
        self.state=state or dict(steps=0,maximum_finite_loss=None,maximum_finite_gradient_norm=None,
            maximum_relative_effective_update=None,spike_thresholds={'loss':100.,'gradient_norm':1e4,'relative_effective_update':10.},
            spikes={'loss':0,'gradient_norm':0,'relative_effective_update':0},
            iterations=[dict(count=0,state_norm=0.,message_norm=0.,relative_message=0.,effective_update_norm=0.,relative_effective_update=0.) for _ in range(3)],
            previous_loss=None,completed_successfully=False)
        self.active=False
        self.context={}
        self.last_loss=None
        self.first_forward=None
        for name,module in model.named_modules():
            if name in ['backbone','pool','classifier'] or name.startswith('communication_blocks.'):
                module.register_forward_hook(self.hook(name))
        block=model.communication_blocks[0]
        block.register_forward_pre_hook(self.before_block)
        block.message_transform.register_forward_hook(self.message)

    def hook(self,name):
        def check(module,args,out):
            if self.active and isinstance(out,torch.Tensor) and not bool(torch.isfinite(out).all()):
                self.first_forward=name
                self.fail('forward',name)
        return check

    def before_block(self,module,args):
        if self.active:
            self.index+=1
            self.norm=args[0].detach().double().flatten(1).norm(dim=1)

    def message(self,module,args,out):
        if not self.active:return
        msg=out.detach().double().flatten(1).norm(dim=1)
        relative=msg/(self.norm+1e-8)
        scale=self.model.residual_scale
        row=self.state['iterations'][self.index]
        row['count']+=len(msg)
        for k,v in dict(state_norm=self.norm,message_norm=msg,relative_message=relative,
                        effective_update_norm=msg*scale,relative_effective_update=relative*scale).items():
            row[k]+=v.sum().item()
        peak=(relative*scale).max().item()
        self.maximum('maximum_relative_effective_update',peak)
        if peak>10:self.state['spikes']['relative_effective_update']+=1

    def maximum(self,key,value):
        if math.isfinite(value):
            old=self.state[key]
            self.state[key]=value if old is None else max(old,value)

    def begin(self,epoch,batch,lr,images,targets):
        self.context=dict(epoch=epoch+1,batch=batch+1,global_step=self.state['steps']+1,learning_rate=lr)
        self.last_loss=None;self.first_forward=None;self.index=-1;self.active=True
        if not bool(torch.isfinite(images).all()):self.fail('input','images')
        if targets.dtype!=torch.int64 or not bool(((targets>=0)&(targets<10)).all()):self.fail('input','labels')

    def loss(self,loss):
        self.last_loss=scalar(loss.detach())
        if not bool(torch.isfinite(loss)):self.fail('loss','cross_entropy')
        value=loss.item();self.maximum('maximum_finite_loss',value)
        if value>100:self.state['spikes']['loss']+=1

    def gradients(self):
        squared=0.;first=None
        for name,p in self.model.named_parameters():
            if p.grad is None:continue
            if not bool(torch.isfinite(p.grad).all()) and first is None:first=name
            squared+=p.grad.detach().double().square().sum().item()
        if first is not None:self.fail('gradient',first)
        norm=math.sqrt(squared);self.maximum('maximum_finite_gradient_norm',norm)
        if norm>1e4:self.state['spikes']['gradient_norm']+=1

    def after_step(self,optimizer):
        for name,p in self.model.named_parameters():
            if not bool(torch.isfinite(p).all()):self.fail('parameter_after_step',name)
            momentum=optimizer.state[p].get('momentum_buffer')
            if momentum is not None and not bool(torch.isfinite(momentum).all()):self.fail('momentum_after_step',name)
        self.state['steps']+=1
        self.state['previous_loss']=self.last_loss
        self.active=False

    def fail(self,stage,name):
        self.active=False
        self.state['failure']={**self.context,'stage':stage,'quantity':name,
            'loss_immediately_before_failure':self.state['previous_loss'],'failure_loss':self.last_loss,
            'first_nonfinite_forward_tensor':self.first_forward,
            'first_module_with_nonfinite_gradients':name.split('.')[0] if stage=='gradient' else None,
            'parameter_scan_order_not_backward_origin':stage=='gradient'}
        self.save()
        raise NumericalFailure(f'{stage}: {name}; {self.context}')

    def save(self,complete=False):
        self.state['completed_successfully']=complete
        write_json(self.run/'stability.json',self.state)
