"""Action-supervision access controls using unchanged recurrent/decoder classes.

The random-history control retains the same frozen RGB frontend, PCA and
pretrained-derived normalizer; it is not an entirely pretraining-free baseline.
"""
from mte.method_names import resolve_control, resolve_visual_arm
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
    arm = resolve_control(arm)
    return 'anchor_'+('solo' if arm.split('_')[1]=='solo' else 'plus')+'_edge_cara_mif'


def fingerprint(items):
    h=hashlib.sha256()
    for k,v in sorted(items.items()):
        a=v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        h.update(k.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def ground(out,arm):
    # Names: mif_{solo,aux}_{pretrained,random}_{frozen,trainable}.
    arm = resolve_control(arm)
    _,composition,initialization,access=arm.split('_')
    assert composition in ('solo','aux') and initialization in ('pretrained','random') and access in ('frozen','trainable')
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
    if initialization=='random':
        C.seed_all(V.SEED)
        if composition=='solo':
            net=FrozenPair(C.RecurrentPolicy(32,dim)).cuda().eval()
        else:
            d1=net.anchor.head[-1].out_features;d2=net.aux.head[-1].out_features
            net=FrozenPair(C.RecurrentPolicy(32,d1),C.RecurrentPolicy(32,d2)).cuda().eval()
    initial_history=fingerprint(net.state_dict())
    train_history=access=='trainable'
    for q in net.parameters():q.requires_grad_(train_history)
    net.train(train_history)
    with torch.no_grad():
        fixed=(net(inputs)[0].reshape(B.BUDGET*200,dim)-mean)/scale
    C.seed_all(V.SEED)
    decoder=C.ActionDecoder(dim,2,hidden=128).cuda()
    initial_decoder=fingerprint(decoder.state_dict())
    parameters=list(decoder.parameters())+(list(net.parameters()) if train_history else [])
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
    assert (initial_history==final_history) if not train_history else (initial_history!=final_history)
    torch.save(dict(state_dict=decoder.cpu().state_dict(),history_state_dict=net.cpu().state_dict(),
        mean=mean.cpu(),scale=scale.cpu(),obs_mean=ck['obs_mean'],obs_std=ck['obs_std'],arm=arm,
        latent_dim=dim,composition=composition,base_arm=base_arm(arm)),out/'decoder.pt')
    np.save(out/'fit_predictions.npy',prediction.cpu().numpy())
    atomic_json(out/'curve.json',curve)
    nh=sum(q.numel() for q in net.parameters());nd=sum(q.numel() for q in decoder.parameters())
    checks=dict(history_initial=initial_history,history_final=final_history,decoder_initial=initial_decoder,
        normalization=fingerprint(dict(mean=mean,scale=scale,obs_mean=ck['obs_mean'],obs_std=ck['obs_std'])),
        batches=batch_hash.hexdigest(),targets=fingerprint(dict(y=y)),inputs=fingerprint(dict(x=inputs)),
        total_history_parameters=nh,decoder_parameters=nd,initialization=initialization,history_trainable=train_history)
    atomic_json(out/'training_identity.json',checks)
    assert not audit['violations']
    return dict(arm=arm,native_action_labels_read=B.BUDGET*200,partner_labels_read=0,simulator_queries=0,
        updates=B.UPDATES,supervised_vectors_per_update=256,total_label_presentations=B.UPDATES*256,
        trainable_parameters=nd+(nh if train_history else 0),frozen_policy_parameters=0 if train_history else nh,
        full_fit_action_mse=fit,stepwise_parity_error=sequence_error,selection='fixed final update; no dev-label/return selection',**checks)


def controller(arm):
    arm = resolve_visual_arm(arm)
    d=torch.load(V.RT/('ground_'+arm)/'decoder.pt',map_location='cuda',weights_only=False)
    net,_=V.pair(d['base_arm']);net.load_state_dict(d['history_state_dict']);net.eval()
    decoder=C.ActionDecoder(d['latent_dim'],2,hidden=128).cuda();decoder.load_state_dict(d['state_dict']);decoder.eval()
    def predict(features,h=None):
        z,h=net((features-torch.as_tensor(d['obs_mean'],device='cuda'))/torch.as_tensor(d['obs_std'],device='cuda'),h)
        return decoder((z-d['mean'])/d['scale']),h
    return predict
