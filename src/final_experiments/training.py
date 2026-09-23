"""New trainer reuses the exact frozen loop via isolated dependency bindings."""
import types
import torch
from src.experiments import phase4_training as frozen
from src.experiments.metadata import write_json,split_metadata as original_split
from src.final_experiments.data import make_loader


class NumericalFailure(RuntimeError):pass


class Monitor:
    def __init__(self,model,run,state=None):
        self.model,self.run=model,run;self.active=False;self.context={};self.loss_value=None
        self.state=state or dict(steps=0,previous_loss=None,completed_successfully=False)
        for name,module in model.named_modules():module.register_forward_hook(self.hook(name))
    def hook(self,name):
        def check(module,args,result):
            if self.active and isinstance(result,torch.Tensor) and not bool(torch.isfinite(result).all()):self.fail('forward',name)
        return check
    def begin(self,epoch,batch,lr,images,targets):
        self.context=dict(epoch=epoch+1,batch=batch+1,global_step=self.state['steps']+1,learning_rate=lr)
        self.active=True;self.loss_value=None
        write_json(self.run/'current_step.json',self.context)
        if not bool(torch.isfinite(images).all()):self.fail('input','images')
    def loss(self,loss):
        value=loss.item();self.loss_value=value if torch.isfinite(loss) else str(value)
        if not bool(torch.isfinite(loss)):self.fail('loss','cross_entropy')
    def gradients(self):
        for name,p in self.model.named_parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):self.fail('gradient',name)
    def after_step(self,optimizer):
        for name,p in self.model.named_parameters():
            if not bool(torch.isfinite(p).all()):self.fail('parameter_after_step',name)
            m=optimizer.state[p].get('momentum_buffer')
            if m is not None and not bool(torch.isfinite(m).all()):self.fail('momentum_after_step',name)
        self.active=False;self.state['steps']+=1;self.state['previous_loss']=self.loss_value
    def fail(self,stage,name):
        self.active=False
        self.state['failure']={**self.context,'stage':stage,'quantity':name,'loss':self.loss_value,'previous_loss':self.state['previous_loss']}
        self.save();raise NumericalFailure(str(self.state['failure']))
    def save(self,complete=False):
        self.state['completed_successfully']=complete;write_json(self.run/'stability.json',self.state)


def train(config,smoke=False,run_dir=None,resume=False,stop_after_epoch=None):
    # Copy the function's globals, never monkeypatch historical modules in place.
    namespace=dict(frozen.train.__globals__)
    from src.final_experiments.evaluation import checked_evaluate
    holder={};loader_phases={}
    def create_monitor(*args,**kwargs):
        monitor=Monitor(*args,**kwargs);holder['monitor']=monitor;return monitor
    def load(*args,**kwargs):
        loader=make_loader(*args,**kwargs)
        loader_phases[id(loader)]=args[1] if len(args)>1 else kwargs['split']
        return loader
    def evaluate(model,loader,device):
        return checked_evaluate(model,loader,device,holder['monitor'],loader_phases.get(id(loader),'evaluation'))
    def split(training,validation,seed,smoke):
        result=original_split(training,validation,seed,smoke)
        if not smoke:result['dataset']=config.get('dataset','CIFAR-10')
        return result
    namespace.update(make_loader=load,StabilityMonitor=create_monitor,split_metadata=split,evaluate=evaluate)
    function=types.FunctionType(frozen.train.__code__,namespace,'train',frozen.train.__defaults__)
    return function(config,smoke,run_dir,resume,stop_after_epoch)
