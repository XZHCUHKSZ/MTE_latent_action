"""Downstream adapters using unchanged scientific recurrent/decoder classes."""
from mte.method_names import resolve_method
from pathlib import Path
import copy
import hashlib
import numpy as np
import torch
from training import visual_supervision_control as visual_control
from training import visual_label_budget as B
from training.composition import restore, FrozenPair
from training.grounding_mpe import train_decoder
from closed_loop_lam_v1 import common as C
from utils.atomic import atomic_json
from utils.access import guard


def visual_base(arm):
    family, composition, _, _ = arm.split('_')
    assert family in ('lapo', 'laom') and composition in ('solo', 'aux')
    return 'anchor_'+('solo' if composition=='solo' else 'plus')+'_'+family+'_state_adapter'


def visual_ground(dest, root, seed, arm, updates):
    B.configure(seed, root, 2, updates=updates)
    visual_control.base_arm = visual_base
    B.V.progress = lambda phase, **kw: atomic_json(dest/'progress.json', dict(phase=phase, **kw))
    return visual_control.ground(dest, arm)


def mpe_net(source, method):
    method = resolve_method(method, "mpe")
    members=method.split('+')
    net, ck=restore(source/'pretrain'/members[0]/'policy/policy.pt')
    if len(members)==2:
        aux, ac=restore(source/'pretrain'/members[1]/'policy/policy.pt')
        assert np.array_equal(ck['obs_mean'],ac['obs_mean']) and np.array_equal(ck['obs_std'],ac['obs_std'])
        net=FrozenPair(net,aux)
    return net,ck


def mpe_ground(dest, source, seed, method, trainable, updates):
    method = resolve_method(method, "mpe")
    allowed=[source/'data/train.npz',source/'data/labels_b32.npz']
    allowed += [source/'pretrain'/m/'policy/policy.pt' for m in method.split('+')]
    audit=guard(dest,allowed,pretraining=False)
    audit['phase']='budgeted_history_finetuning' if trainable else 'frozen_path_parity'
    try:
        with np.load(allowed[0]) as f:
            assert set(f.files)=={'positions'};x=f['positions'][:32,:23]
        with np.load(allowed[1]) as f:
            assert set(f.files)=={'actions','local_ids'} and np.array_equal(f['local_ids'],np.arange(32))
            actions=f['actions'];assert actions.shape==(32,22,2)
        net,ck=mpe_net(source,method)
        inputs=torch.tensor((x-ck['obs_mean'])/ck['obs_std'])
        with torch.no_grad():z=net(inputs)[0][:,1:].numpy()
        initial_history=visual_control.fingerprint(net.state_dict())
        val=np.arange(0,32,4);fit=np.setdiff1d(np.arange(32),val)
        C.seed_all(seed);decoder=C.ActionDecoder(z.shape[-1],2)
        initial_decoder=visual_control.fingerprint(decoder.state_dict())
        rng=np.random.default_rng(seed+211);bh=hashlib.sha256()
        for _ in range(updates):bh.update(rng.integers(0,len(fit)*22,256).tobytes())
        if not trainable:
            decoder,diag=train_decoder(z,actions,fit,val,seed,updates)
        else:
            net=net.cuda();decoder=decoder.cuda()
            for q in net.parameters():q.requires_grad_(True)
            net.train(True);inputs=inputs.cuda();y=torch.tensor(actions,device='cuda')
            opt=torch.optim.Adam(list(decoder.parameters())+list(net.parameters()),lr=.001)
            rng=np.random.default_rng(seed+211);best=float('inf');best_net=None;best_decoder=None;selected=0
            for step in range(1,updates+1):
                ix=rng.integers(0,len(fit)*22,256)
                latent=net(inputs[fit])[0][:,1:].reshape(-1,z.shape[-1])
                loss=(decoder(latent[ix])-y[fit].reshape(-1,2)[ix]).square().mean()
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();opt.step()
                if step==1 or step%25==0 or step==updates:
                    net.eval()
                    with torch.no_grad():candidate=(decoder(net(inputs[val])[0][:,1:])-y[val]).square().mean().item()
                    net.train(True)
                    if candidate<best:
                        best=candidate;selected=step
                        best_net={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                        best_decoder={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
                    atomic_json(dest/'progress.json',dict(update=step,updates=updates))
            net.load_state_dict(best_net);decoder.load_state_dict(best_decoder)
            net=net.cpu().eval();decoder=decoder.cpu().eval();inputs=inputs.cpu()
            diag=dict(validation_action_mse=best,selected_update=selected,updates=updates)
        with torch.no_grad():
            predictions=decoder(net(inputs.cpu())[0][:,1:]).numpy()
            changed=inputs.cpu().clone();changed[:,13:]+=13
            causal=float((net(inputs.cpu())[0][:,:13]-net(changed)[0][:,:13]).abs().max())
        assert causal==0
        final_history=visual_control.fingerprint(net.state_dict())
        assert (initial_history!=final_history) if trainable else (initial_history==final_history)
        torch.save(dict(history=net.state_dict(),decoder=decoder.state_dict(),mean=ck['obs_mean'],std=ck['obs_std'],method=method,budget=32),dest/'decoder.pt')
        np.save(dest/'fit_predictions.npy',predictions)
        identity=dict(history_initial=initial_history,history_final=final_history,decoder_initial=initial_decoder,
            normalization=visual_control.fingerprint({'mean':ck['obs_mean'],'std':ck['obs_std']}),
            targets=visual_control.fingerprint({'actions':actions}),inputs=visual_control.fingerprint({'inputs':inputs.cpu()}),
            batches=bh.hexdigest(),fit_ids=fit.tolist(),validation_ids=val.tolist(),
            history_parameters=sum(q.numel() for q in net.parameters()),decoder_parameters=sum(q.numel() for q in decoder.parameters()))
        atomic_json(dest/'training_identity.json',identity)
        assert not audit['violations']
        return dict(arm=method,grounding=diag,native_action_labels_read=704,fit_action_labels=528,
            validation_action_labels=176,partner_labels_read=0,simulator_queries=0,history_trainable=trainable,
            future_prefix_error=causal,**identity)
    finally:
        atomic_json(dest/'access_audit.json',audit)
