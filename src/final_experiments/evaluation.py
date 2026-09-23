"""Finite-value validation guards outside the frozen forward timing interval."""
import math
import types
import torch
from src.evaluation import evaluator
from src.evaluation.metrics import ClassificationMetrics
from src.experiments.metadata import write_json


def tensor_summary(tensor):
    t=tensor.detach()
    finite=torch.isfinite(t)
    values=t[finite].double()
    return dict(shape=list(t.shape),dtype=str(t.dtype),finite=bool(finite.all()),
        nan_count=int(torch.isnan(t).sum()),inf_count=int(torch.isinf(t).sum()),
        finite_min=values.min().item() if values.numel() else None,
        finite_max=values.max().item() if values.numel() else None)


def checked_evaluate(model,loader,device,monitor,phase='validation'):
    """Reuse original evaluation including warm-up, timing and metric arithmetic.

    Only invalid predictions trigger an additional eval forward to localize the
    module. Healthy forwards have no extra per-module checks inside timed regions.
    """
    monitor.active=False
    batch_images=None
    base_context=dict(monitor.context)
    def fail(quantity,tensor=None):
        detail=tensor_summary(tensor) if tensor is not None else {}
        first=[];handles=[]
        class Located(Exception):pass
        def hook(name):
            def check(module,args,out):
                if isinstance(out,torch.Tensor) and not bool(torch.isfinite(out).all()):
                    first.append(dict(module=name,output=tensor_summary(out)))
                    raise Located()
            return check
        try:
            if batch_images is not None:
                for name,module in model.named_modules():handles.append(module.register_forward_hook(hook(name)))
                # Restore RNG after localization; the scientific eval path is unchanged.
                devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    try:model(batch_images.to(device))
                    except Located:pass
        finally:
            for handle in handles:handle.remove()
        monitor.state['evaluation_evidence']=dict(phase=phase,quantity=quantity,detail=detail,
            first_nonfinite_module=first[0] if first else None,
            nonfinite_model_state={k:tensor_summary(v) for k,v in model.state_dict().items() if not bool(torch.isfinite(v).all())},
            localization_note='Extra eval forward on the same batch after detecting invalid scoring output; not a new training step.')
        monitor.loss_value=None
        monitor.fail(phase+'_'+quantity,first[0]['module'] if first else quantity)
    class Metrics(ClassificationMetrics):
        def update(self,logits,targets):
            if logits.ndim!=2:raise ValueError('Classification logits must be two-dimensional')
            if not bool(torch.isfinite(logits).all()):fail('forward',logits)
            # Match the existing float32 metric conversion; guard conversion overflow.
            converted=logits.detach().float().cpu()
            if not bool(torch.isfinite(converted).all()):fail('metric_conversion',converted)
            probabilities=converted.softmax(1)
            if not bool(torch.isfinite(probabilities).all()):fail('probabilities',probabilities)
            super().update(logits,targets)
            if not all(math.isfinite(v) for v in [self.loss,self.brier]):fail('metric_accumulation')
    class TrackedLoader:
        def __iter__(self):
            nonlocal batch_images
            for index,(images,targets) in enumerate(loader,1):
                batch_images=images
                monitor.context=dict(epoch=base_context.get('epoch'),phase=phase,evaluation_batch=index,
                    last_completed_training_batch=base_context.get('batch'),
                    global_step=monitor.state['steps'],learning_rate=base_context.get('learning_rate'))
                write_json(monitor.run/'current_evaluation.json',monitor.context)
                if not bool(torch.isfinite(images).all()):fail('input',images)
                yield images,targets
    original=evaluator.evaluate.__wrapped__
    namespace=dict(original.__globals__);namespace['ClassificationMetrics']=Metrics
    fn=types.FunctionType(original.__code__,namespace,'evaluate',original.__defaults__)
    with torch.inference_mode():result=fn(model,TrackedLoader(),device)
    monitor.context=base_context
    return result
