"""Explicit CPU call adapter for the author's bottom-right causal convention.

The vendor CPU fallback uses an upper-left triangular mask for rectangular
queries. Translate causal=True to an explicit bottom-right additive mask, then
use its existing noncausal-with-bias path. Fully masked rows are explicitly
zeroed before the original output projection, matching Flash/SDPA semantics. No weights or vendor source change.
"""
import torch


def bottom_right_causal_call(module,args,kwargs):
    module.proj._cpu_empty_prefix=0
    if not kwargs.get('causal',False):return args,kwargs
    query=args[0]
    context=args[1] if len(args)>1 else kwargs.get('x_k')
    n=query.shape[-2];m=n if context is None else context.shape[-2]
    if m<1:raise ValueError('Empty key sequences are unsupported')
    module.proj._cpu_empty_prefix=max(0,n-m)
    if kwargs.get('attn_bias') is not None:raise ValueError('Do not combine causal and supplied bias')
    row=torch.arange(n,device=query.device)[:,None]
    col=torch.arange(m,device=query.device)[None,:]
    allowed=col <= row+(m-n)
    # Avoid NaN softmax for empty rows; erase their value before output projection.
    if n>m:allowed[:n-m,0]=True
    bias=torch.zeros((n,m),device=query.device,dtype=query.dtype).masked_fill(~allowed,float('-inf'))
    kwargs=dict(kwargs,causal=False,attn_bias=bias)
    return args,kwargs


def zero_empty_rows(module,args):
    prefix=getattr(module,'_cpu_empty_prefix',0)
    if not prefix:return args
    x=args[0]
    mask=torch.ones(x.shape[-2],device=x.device,dtype=x.dtype)
    mask[:prefix]=0
    return (x*mask[:,None],)+args[1:]


def attach(module):
    return [module.register_forward_pre_hook(bottom_right_causal_call,with_kwargs=True),
            module.proj.register_forward_pre_hook(zero_empty_rows)]


def install(model):
    from models.modules.attention.xformer_attn import xFromersAttention,XFORMERS_ENABLED
    from models.modules.attention.flash_attn import FlashAttention
    if XFORMERS_ENABLED or any(isinstance(m,FlashAttention) for m in model.modules()):
        raise ValueError('CPU adapter requires author xformers-disabled modules exclusively')
    return [handle for m in model.modules() if isinstance(m,xFromersAttention) for handle in attach(m)]


def verify():
    """Compare forward/input/parameter gradients with independent PyTorch SDPA."""
    from copy import deepcopy
    from configs.models.modules.attention.attn_cfg import AttnConfig
    from models.modules.attention.xformer_attn import xFromersAttention
    rows=[]
    for n,m in [(4,4),(1,2),(5,6),(2,1),(5,4),(5,2)]:
        torch.manual_seed(7751+n)
        module=xFromersAttention(AttnConfig(d_model=32,num_heads=4,is_self_attn=False)).double()
        attach(module)
        reference=deepcopy(module);reference._forward_pre_hooks.clear();reference.proj._forward_pre_hooks.clear()
        q=torch.randn(2,n,32,dtype=torch.float64,requires_grad=True)
        k=torch.randn(2,m,32,dtype=torch.float64,requires_grad=True)
        rq=q.detach().clone().requires_grad_();rk=k.detach().clone().requires_grad_()
        actual=module(q,k,causal=True)
        query=reference.q(rq).reshape(2,n,4,8).transpose(1,2)
        kv=reference.kv(rk).reshape(2,m,2,4,8).permute(2,0,3,1,4)
        # Construct the mask independently by aligning sequence ends.
        mask=torch.tensor([[j-m<=i-n for j in range(m)] for i in range(n)],dtype=torch.bool)
        y=torch.nn.functional.scaled_dot_product_attention(query,kv[0],kv[1],attn_mask=mask,dropout_p=0.)
        expected=reference.proj(y.transpose(1,2).reshape(2,n,32))
        weight=torch.randn_like(actual)
        (actual*weight).sum().backward();(expected*weight).sum().backward()
        errors=[float((actual-expected).abs().max()),float((q.grad-rq.grad).abs().max()),float((k.grad-rk.grad).abs().max())]
        errors.extend(float((v.grad-dict(reference.named_parameters())[name].grad).abs().max()) for name,v in module.named_parameters())
        if max(errors)>1e-10:raise AssertionError(errors)
        rows.append(dict(query_length=n,key_length=m,max_forward_gradient_error=max(errors)))
    return rows
