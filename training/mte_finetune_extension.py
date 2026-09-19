"""Budgeted downstream adaptation; original representations stay immutable."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from training.composition import restore, FrozenPair
from training import visual_supervision_control as S
from training import visual_label_budget as B
from training.grounding_mamujoco import train_decoder, _prefix_mask, _labels, _loss
from utils.access import guard
from utils.atomic import atomic_json


def visual_base(arm):
    family, comp, initialization, access = arm.split('_')
    assert family in ('simple', 'graph', 'tree') and comp in ('solo', 'aux')
    return 'anchor_'+('solo' if comp=='solo' else 'plus')+'_edge_cara_mobius_'+family


def visual_ground(dest, root, seed, arm, updates):
    B.configure(seed, root, 2, updates=updates)
    S.base_arm = visual_base
    B.V.progress = lambda phase, **kw: atomic_json(dest/'progress.json', dict(phase=phase, **kw))
    return S.ground(dest, arm)


def mamujoco_net(base, auxiliary):
    a, ck = restore(base); b, bc = restore(auxiliary)
    assert np.array_equal(ck['obs_mean'], bc['obs_mean'])
    assert np.array_equal(ck['obs_std'], bc['obs_std'])
    return FrozenPair(a, b).eval(), ck


def mamujoco_ground(dest, spec, trainable, updates):
    allowed=[spec[k] for k in ['train','labels','base','auxiliary']]
    audit=guard(dest,allowed,pretraining=False)
    audit['phase']='budgeted_history_finetuning' if trainable else 'frozen_parity'
    try:
        with np.load(spec['train']) as f:
            assert set(f.files)=={'observations','valid_mask'}
            x=f['observations'][:8].astype(np.float32)
            mask=_prefix_mask(f['valid_mask'][:8],8,200)
        with np.load(spec['labels']) as f:
            assert set(f.files)=={'actions','local_ids'} and np.array_equal(f['local_ids'],np.arange(8))
            actions=_labels(f['actions'],(8,200,2))
        net,ck=mamujoco_net(spec['base'],spec['auxiliary'])
        inputs=torch.tensor((x[:,:200]-ck['obs_mean'])/ck['obs_std'])
        with torch.no_grad():z=net(inputs)[0].numpy();z[~mask]=0
        ih=S.fingerprint(net.state_dict());ds=spec['decoder_seed']
        fit=np.array([1,2,3,5,6,7]);val=np.array([0,4])
        C.seed_all(ds);decoder=C.ActionDecoder(z.shape[-1],2)
        di=S.fingerprint(decoder.state_dict())
        rng=np.random.default_rng(ds+211);bh=hashlib.sha256();nfit=int(mask[fit].sum())
        for _ in range(updates):bh.update(rng.integers(0,nfit,min(512,nfit)).tobytes())
        if not trainable:
            decoder,diag=train_decoder(z,actions,mask,fit,val,ds,updates)
        else:
            net=net.cuda();decoder=decoder.cuda();inputs=inputs.cuda()
            for q in net.parameters():q.requires_grad_(True)
            net.train(True)
            y=torch.tensor(actions,device='cuda');m=torch.tensor(mask,device='cuda')
            opt=torch.optim.Adam(list(net.parameters())+list(decoder.parameters()),lr=.001)
            rng=np.random.default_rng(ds+211);best=float('inf');selected=0
            for step in range(1,updates+1):
                ix=rng.integers(0,nfit,min(512,nfit))
                latent=net(inputs[fit])[0][m[fit]]
                loss=(decoder(latent[ix])-y[fit][m[fit]][ix]).square().mean()
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();opt.step()
                if step==1 or step%25==0 or step==updates:
                    net.eval()
                    with torch.no_grad():score=_loss(decoder(net(inputs[val])[0]),y[val],m[val]).item()
                    net.train(True)
                    if score<best:
                        best=score;selected=step
                        best_net={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                        best_decoder={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
                    atomic_json(dest/'progress.json',dict(update=step,updates=updates))
            net.load_state_dict(best_net);decoder.load_state_dict(best_decoder)
            net=net.cpu().eval();decoder=decoder.cpu().eval();inputs=inputs.cpu()
            diag=dict(validation_action_mse=best,selected_update=selected,updates=updates)
        with torch.no_grad():
            values=net(inputs)[0];changed=inputs.clone();changed[:,100:]+=19
            causal=float((values[:,:100]-net(changed)[0][:,:100]).abs().max())
            step_values=[];h=None
            for i in range(200):
                v,h=net(inputs[:,i:i+1],h);step_values.append(v)
            sequential=float((torch.cat(step_values,1)-values).abs().max())
            predictions=decoder(values).numpy()
        assert causal==0 and sequential<=1e-5,(causal,sequential)
        final=S.fingerprint(net.state_dict())
        assert ih!=final if trainable else ih==final
        torch.save(dict(history=net.state_dict(),decoder=decoder.state_dict(),mean=ck['obs_mean'],std=ck['obs_std'],budget=8,spec=spec),dest/'decoder.pt')
        np.save(dest/'fit_predictions.npy',predictions)
        identity=dict(history_initial=ih,history_final=final,decoder_initial=di,
            normalization=S.fingerprint(dict(mean=ck['obs_mean'],std=ck['obs_std'])),
            inputs=S.fingerprint(dict(x=inputs,mask=mask)),targets=S.fingerprint(dict(actions=actions)),
            batches=bh.hexdigest(),fit_ids=fit.tolist(),validation_ids=val.tolist(),
            history_parameters=sum(q.numel() for q in net.parameters()),decoder_parameters=sum(q.numel() for q in decoder.parameters()))
        atomic_json(dest/'training_identity.json',identity)
        assert not audit['violations']
        return dict(native_action_labels_read=int(mask.sum()),padded_action_slots=1600,
            fit_action_labels=nfit,validation_action_labels=int(mask[val].sum()),partner_labels_read=0,
            simulator_queries=0,grounding=diag,history_trainable=trainable,future_prefix_error=causal,
            sequential_error=sequential,**identity)
    finally:atomic_json(dest/'access_audit.json',audit)
