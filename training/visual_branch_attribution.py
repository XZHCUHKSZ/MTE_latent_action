"""Branch-selective downstream adaptation of unchanged recurrent/decoder classes.
Derived from the accepted visual_supervision_control trainer: only gradient
access differs; exact frozen and joint reproduction gates precede evaluation.
"""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from training import visual_label_budget as B
from training.composition import FrozenPair
from closed_loop_lam_v1 import common as C
from utils.atomic import atomic_json
V = B.V


def base_arm(arm):
    family,composition,initialization,access=arm.split('_')
    assert family in ('simple','graph','tree','mif','laom') and composition=='aux' and initialization=='pretrained'
    method=('edge_cara_mobius_'+family) if family in ('simple','graph','tree') else ('edge_cara_mif' if family=='mif' else 'laom_state_adapter')
    return 'anchor_plus_'+method


def fingerprint(items):
    h=hashlib.sha256()
    for k,v in sorted(items.items()):
        a=v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        h.update(k.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def ground(out,arm):
    # Names: mif_{solo,aux}_{pretrained,random}_{frozen,trainable}.
    _,composition,initialization,access=arm.split('_')
    assert composition in ('solo','aux') and initialization=='pretrained' and access in ('frozen','baseonly','auxonly','trainable')
    files=[B.SOURCE/'features/features.npz']+[B.LABELS/f'{i:04d}.npy' for i in range(B.BUDGET)]
    audit=V.watch(out,files+V.policy_paths(),'supervision_access_control')
    x=np.load(files[0])['x'][:,:,0]
    y=torch.tensor(np.stack([np.load(f) for f in files[1:]]),device='cuda')
    net,ck=V.pair(base_arm(arm))
    inputs=torch.tensor((x[:B.BUDGET,:200]-ck['obs_mean'])/ck['obs_std'],device='cuda')
    # Shared normalizer is fixed before any action updates, including scratch arms.
    with torch.no_grad():
        values=net(inputs)[0].reshape(B.BUDGET*200,-1)
        mean=values.mean(0).detach();scale=values.std(0).clamp_min(.0001).detach()
    dim=values.shape[-1]
    initial_history=fingerprint(net.state_dict())
    train_history=access!='frozen'
    train_base=access in ('baseonly','trainable');train_aux=access in ('auxonly','trainable')
    branch_initial={k:fingerprint(m.state_dict()) for k,m in [('base',net.anchor),('aux',net.aux)]}
    for q in net.anchor.parameters():q.requires_grad_(train_base)
    for q in net.aux.parameters():q.requires_grad_(train_aux)
    net.train(train_history)
    with torch.no_grad():
        fixed=(net(inputs)[0].reshape(B.BUDGET*200,dim)-mean)/scale
    C.seed_all(V.SEED)
    decoder=C.ActionDecoder(dim,2,hidden=128).cuda()
    initial_decoder=fingerprint(decoder.state_dict())
    parameters=list(decoder.parameters())+[q for q in net.parameters() if q.requires_grad]
    opt=torch.optim.Adam(parameters,lr=.001)
    rng=np.random.default_rng(V.SEED);batch_hash=hashlib.sha256();curve=[]
    for step in range(B.UPDATES):
        ix_np=rng.integers(B.BUDGET*200,size=256);batch_hash.update(ix_np.tobytes())
        ix=torch.tensor(ix_np,device='cuda')
        values=(net(inputs)[0].reshape(-1,dim)-mean)/scale if train_history else fixed
        loss=F.mse_loss(decoder(values[ix]),y.reshape(-1,2)[ix])
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0 or step==B.UPDATES-1:
            curve.append(dict(update=step+1,loss=float(loss.detach())))
            V.progress('history_action_grounding',arm=arm,update=step+1,updates=B.UPDATES)
    net.eval();decoder.eval()
    with torch.no_grad():
        prediction=decoder((net(inputs)[0]-mean)/scale)
        fit=float(F.mse_loss(prediction,y))
        hidden=None;steps=[]
        for t in range(200):
            z,hidden=net(inputs[:,t:t+1],hidden);steps.append(decoder((z-mean)/scale))
        sequence_error=float((torch.cat(steps,1)-prediction).abs().max())
        assert sequence_error<1e-5,sequence_error
    final_history=fingerprint(net.state_dict())
    branch_final={k:fingerprint(m.state_dict()) for k,m in [('base',net.anchor),('aux',net.aux)]}
    assert (branch_initial['base']!=branch_final['base'])==train_base
    assert (branch_initial['aux']!=branch_final['aux'])==train_aux
    assert (initial_history==final_history) if not train_history else (initial_history!=final_history)
    torch.save(dict(state_dict=decoder.cpu().state_dict(),history_state_dict=net.cpu().state_dict(),
        mean=mean.cpu(),scale=scale.cpu(),obs_mean=ck['obs_mean'],obs_std=ck['obs_std'],arm=arm,
        latent_dim=dim,composition=composition,base_arm=base_arm(arm)),out/'decoder.pt')
    np.save(out/'fit_predictions.npy',prediction.cpu().numpy())
    atomic_json(out/'curve.json',curve)
    nh=sum(q.numel() for q in net.parameters());nd=sum(q.numel() for q in decoder.parameters())
    checks=dict(branch_initial=branch_initial,branch_final=branch_final,base_trainable=train_base,aux_trainable=train_aux,
        base_parameters=sum(q.numel() for q in net.anchor.parameters()),aux_parameters=sum(q.numel() for q in net.aux.parameters()),history_initial=initial_history,history_final=final_history,decoder_initial=initial_decoder,
        normalization=fingerprint(dict(mean=mean,scale=scale,obs_mean=ck['obs_mean'],obs_std=ck['obs_std'])),
        batches=batch_hash.hexdigest(),targets=fingerprint(dict(y=y)),inputs=fingerprint(dict(x=inputs)),
        total_history_parameters=nh,decoder_parameters=nd,initialization=initialization,history_trainable=train_history)
    atomic_json(out/'training_identity.json',checks)
    assert not audit['violations']
    return dict(arm=arm,native_action_labels_read=B.BUDGET*200,partner_labels_read=0,simulator_queries=0,
        updates=B.UPDATES,supervised_vectors_per_update=256,total_label_presentations=B.UPDATES*256,
        trainable_parameters=nd+sum(q.numel() for q in net.parameters() if q.requires_grad),frozen_policy_parameters=sum(q.numel() for q in net.parameters() if not q.requires_grad),
        full_fit_action_mse=fit,stepwise_parity_error=sequence_error,selection='fixed final update; no dev-label/return selection',**checks)

