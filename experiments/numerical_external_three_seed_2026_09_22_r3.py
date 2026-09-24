"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import os

os.environ['CUDA_VISIBLE_DEVICES']=''

os.environ['XFORMERS_DISABLED']='1'

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'

from copy import deepcopy

from dataclasses import asdict

import hashlib

import json

from pathlib import Path

import shutil

import sys

import sysconfig

import time

import traceback

P=Path(__file__).resolve().parents[1];W=P.parents[1]

P=Path(__file__).resolve().parents[1];W=P.parents[1]

sys.path.insert(0,str(P))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=True,indent=2,default=str),encoding='utf-8');q.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def local(p):
    s=str(p).replace('\\','/')
    if sys.platform=='linux' and s[1:3]==':/':return Path('/mnt/'+s[0].lower()+'/'+s[3:])
    if sys.platform=='win32' and s.startswith('/mnt/d/'):return Path('D:/'+s[7:])
    return Path(s)

def verify(m):
    for p,h in m.items():assert sha(local(p))==h,p

def sources(env,seed):
    if env=='mpe':
        root=P/f'outputs/mpe_temporal_full_scale_5seeds/seed{seed}/data'
        return root/'train.npz',root/'dev.npz',root/'labels_b32.npz'
    root=W/'observation_scale_restore_2026_09_10/runtime'
    return root/f'mamujoco_s{seed}_n90__prepare/train.npz',root/f'mamujoco_s{seed}_n90__prepare/dev.npz',root/f'mamujoco_s{seed}_n90__export_32/labels.npz'

def identity(env,seed,method):return f'{env}_s{seed}_{method}'

def bank(env,seed,qualification):
    from training.observation_only_sequence_bank import ObservationSequenceBank as Bank
    tr,dv,_=sources(env,seed);key='positions' if env=='mpe' else 'observations'
    a=Bank.load(tr,key);b=Bank.load(dv,key)
    if qualification:
        a=Bank(a.observations[:2],a.valid_transitions[:2]);b=Bank(b.observations[:1],b.valid_transitions[:1])
    return a,b

def artifact_guard(dest,allowed,stage):
    allowed={Path(p).resolve() for p in allowed};audit=dict(stage=stage,numeric_reads=[],violations=[])
    def hook(event,args):
        if event=='socket.connect':raise PermissionError('Network denied')
        if stage!='evaluate' and event=='import' and str(args[0]).split('.')[0] in {'mujoco','mujoco_py','pettingzoo','mpe2'}:
            audit['violations'].append(str(args[0]));raise PermissionError('Simulator denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() not in {'.npy','.npz','.pt','.pth','.pkl','.pickle','.zip','.h5','.hdf5'}:return
        writing=bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        # CPython probes an optional, absent standard-library archive. Its exact
        # directory differs between Windows and Linux; never exempt a real file.
        archive=f'python{sys.version_info.major}{sys.version_info.minor}.zip'
        runtime_archives={(Path(sys.base_prefix)/archive).resolve(),(Path(sysconfig.get_path('stdlib')).parent/archive).resolve()}
        if not writing and p in runtime_archives and str(p) in {str(Path(q).resolve()) for q in sys.path if q} and not p.exists():return
        if not writing:audit['numeric_reads'].append(str(p))
        if not p.is_relative_to(dest.resolve()) and (writing or p not in allowed):
            audit['violations'].append(str(p));raise PermissionError('Artifact outside stage allowlist: '+str(p))
    sys.addaudithook(hook)
    if stage!='evaluate':
        import gymnasium
        def forbidden(*args,**kwargs):raise PermissionError('No simulator construction in training')
        gymnasium.make=forbidden
    return audit

def construct(method,contract,seed,dest):
    import torch
    if method=='otf':
        from omegaconf import OmegaConf
        vendor=P/'third_party/otf_lam_official';sys.path.insert(0,str(vendor))
        from training.otf_state_adapter import OTFStateVQ,OTFStateLAM
        cfg=OmegaConf.create(OmegaConf.to_container(OmegaConf.load(vendor/'configs/otf_vqvae/walker.yaml').model,resolve=True))
        pixel=read(P/'configs/external_otf_pixel.json')
        tok=OTFStateVQ(contract.observation_dim,cfg)
        return tok,lambda:OTFStateLAM(tok,pixel),None
    vendor=P/'third_party/flam_source_complete_1c707be'
    sys.path[:0]=[str(vendor),str(vendor/'third_party_models/cosmos'),str(P/'experiments')]
    from experiments.external_flam_config import author_config
    from training.flam_state_adapter import NumericalTokenizer,factored_model
    from training.flam_state_cpu_attention_r3 import install
    assets=P/'provenance/flam_perceptual_assets_1c707be_2026_09_21'
    torch.hub.set_dir(str(assets/'torch_hub'))
    import models.modules.loss.lpips as LP
    LP.REPO_PATH=dest/'perceptual_path_adapter'
    lp=LP.REPO_PATH/'models/modules/loss/lpips.pth';lp.parent.mkdir(parents=True);shutil.copyfile(assets/'lpips.pth',lp)
    c=read(P/'configs/external_flam_author.json');c['seed']=seed
    cfg=author_config(c);cfg.lam.attn_type='xformers';cfg.lam.__post_init__();cfg.__post_init__()
    save(dest/'effective_author_config.json',asdict(cfg));tok=NumericalTokenizer(cfg,contract.observation_dim)
    def build():
        model=factored_model(cfg,tok);install(model);return model
    return tok,build,cfg

def pre(c,env,seed,method,u,dest,progress):
    assert env=='mpe', 'Only the manuscript MPE cohort is exposed'
    import numpy as np
    import torch
    from training.external_state_interfaces import StateContract,StateHistoryController
    contract=StateContract.create(env,method);tr,dv=bank(env,seed,c['qualification'])
    mean,scale=tr.fit_normalization();idx=contract.window_index(tr);rng=np.random.default_rng(seed)
    tok,build,_=construct(method,contract,seed,dest)
    opt=torch.optim.Adam(tok.parameters(),lr=1e-4);losses=[];batch=c['sequence_batch'];warmup=c['vq_warmup']
    def sample():return torch.from_numpy(contract.take(tr,idx[rng.integers(len(idx),size=batch)],mean,scale))
    for step in range(c['frontend_updates']):
        seq=sample()
        if method=='otf':
            current=seq[:,:-1].reshape(-1,contract.observation_dim);future=seq[:,1:].reshape_as(current)
            if step==warmup:tok.initialize(current,future,seed)
            r=tok(current,future,quantized=step>=warmup);loss=r['loss']
        else:loss,_=tok(seq)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(tok.parameters(),1.);opt.step()
        if method=='otf' and step>=warmup:tok.quantizer.update_codebook(r['tokens'],r['indices'],step+1)
        if step%50==0 or step+1==c['frontend_updates']:
            losses.append([step+1,float(loss.detach())]);progress('frontend',completed=step+1,total=c['frontend_updates'])
    torch.save(dict(state_dict=tok.state_dict(),updates=c['frontend_updates'],qualification=c['qualification']),dest/'frontend.pt')
    model=build();frozen={k:v.clone() for k,v in (model.vq.state_dict() if method=='otf' else model.state_dict()).items() if method=='otf' or k.startswith(('encoder.','decoder.','tokenizer_quantizer.'))}
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=1e-4);rep_losses=[]
    if method=='flam':from training.flam_state_adapter import training_loss,transition_code
    for step in range(c['representation_updates']):
        seq=sample();model.train()
        loss=model(seq[:,:-1].reshape(-1,contract.observation_dim),seq[:,1:].reshape(-1,contract.observation_dim))['loss'] if method=='otf' else training_loss(model,seq)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        if step%50==0 or step+1==c['representation_updates']:
            rep_losses.append([step+1,float(loss.detach())]);progress('representation',completed=step+1,total=c['representation_updates'])
    state=model.vq.state_dict() if method=='otf' else model.state_dict()
    assert all(torch.equal(v,state[k]) for k,v in frozen.items())
    torch.save(dict(state_dict=model.state_dict(),updates=c['representation_updates'],qualification=c['qualification']),dest/'representation.pt')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    def encode(x):return model(x[:,-2],x[:,-1])['z_act'] if method=='otf' else transition_code(model,x)
    # Group only equal-length causal contexts, preserving every valid transition.
    exports=[];parity=[]
    for split,b in [('train',tr),('dev',dv)]:
        valid=contract.valid_target_mask(b);times=contract.control_times(b)
        z=np.zeros((*valid.shape,contract.latent_dim),np.float32);groups={}
        for e,ti in np.argwhere(valid):
            frames=contract.export_frames(int(times[ti]));groups.setdefault(len(frames),[]).append((int(e),int(ti),frames))
        done=0
        with torch.no_grad():
            for rows in groups.values():
                for lo in range(0,len(rows),c['export_batch']):
                    chunk=rows[lo:lo+c['export_batch']]
                    x=torch.tensor(np.stack([(b.observations[e,f].astype(np.float64)-mean)/scale for e,ti,f in chunk]),dtype=torch.float32)
                    zz=encode(x);assert torch.isfinite(zz).all()
                    if lo==0:
                        err=float((zz[:1]-encode(x[:1])).abs().max());assert err<.001,err;parity.append(err)
                    for row,value in zip(chunk,zz.numpy()):z[row[0],row[1]]=value
                    done+=len(chunk)
                    if done%256<c['export_batch'] or done==int(valid.sum()):progress('export_'+split,completed=done,total=int(valid.sum()))
        exports.append(z)
    np.savez_compressed(dest/'latents.npz',train=exports[0],dev=exports[1])
    # Same shared GRU, observation-only train normalization, observation-dev selection.
    net=StateHistoryController(contract,mean,scale);zmean=exports[0][contract.valid_target_mask(tr)].mean(0);zstd=exports[0][contract.valid_target_mask(tr)].std(0).clip(1e-6)
    opt=torch.optim.Adam(net.history.parameters(),lr=.001);hrng=np.random.default_rng(seed+101)
    tx=torch.tensor(contract.history_observations(tr));dx=torch.tensor(contract.history_observations(dv))
    tz=torch.tensor((exports[0]-zmean)/zstd);dz=torch.tensor((exports[1]-zmean)/zstd)
    tm=torch.tensor(contract.valid_target_mask(tr));dm=torch.tensor(contract.valid_target_mask(dv));curve=[];best=float('inf');selected=0;best_state=None
    for step in range(1,c['history_updates']+1):
        ix=hrng.choice(len(tx),min(16,len(tx)),replace=False)
        pred=contract.select_history_outputs(net.history((tx[ix]-net.obs_mean)/net.obs_scale)[0])
        loss=(pred-tz[ix]).square()[tm[ix]].mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%100==0 or step==c['history_updates']:
            with torch.no_grad():
                err=[]
                for lo in range(0,len(dx),8):
                    p=contract.select_history_outputs(net.history((dx[lo:lo+8]-net.obs_mean)/net.obs_scale)[0])
                    err.append((p-dz[lo:lo+8]).square()[dm[lo:lo+8]].reshape(-1))
                score=float(torch.cat(err).mean());assert np.isfinite(score)
            if score<best:best=score;selected=step;best_state=deepcopy(net.history.state_dict())
            curve.append([step,float(loss.detach()),score]);progress('history',completed=step,total=c['history_updates'])
    net.history.load_state_dict(best_state);torch.save(dict(state_dict=net.state_dict(),contract=asdict(contract),qualification=c['qualification'],native_action_labels_read=0,updates=c['history_updates'],selected_update=selected),dest/'history.pt')
    return dict(passed=True,qualification=c['qualification'],contract=asdict(contract),frontend_updates=c['frontend_updates'],representation_updates=c['representation_updates'],history_updates=c['history_updates'],frontend_curve=losses,representation_curve=rep_losses,history_curve=curve,tokenizer_frozen=True,export_updates=0,batch_parity=parity,native_action_labels_read=0,simulator_queries=0,train_episodes=len(tx),dev_episodes=len(dx),representation_parameters=sum(p.numel() for p in model.parameters()),representation_trainable_parameters=sum(p.numel() for p in params))

def restore(u):
    import torch
    import numpy as np
    from training.external_state_interfaces import StateContract,StateHistoryController
    ck=torch.load(u/'pre/history.pt',map_location='cpu',weights_only=False);co=StateContract(**ck['contract'])
    net=StateHistoryController(co,np.zeros(co.observation_dim),np.ones(co.observation_dim));net.load_state_dict(ck['state_dict'],strict=True);return net

def ground(c,env,seed,method,u,dest,progress):
    assert env=='mpe', 'Only the manuscript MPE cohort is exposed'
    import numpy as np
    import torch
    from closed_loop_lam_v1.common import ActionDecoder
    from training.visual_supervision_control import fingerprint
    tr,_=bank(env,seed,c['qualification']);net=restore(u).eval();co=net.contract;b=c['budget'];labels=sources(env,seed)[2]
    with np.load(labels,allow_pickle=False) as f:
        assert set(f.files)=={'actions','local_ids'} and np.array_equal(f['local_ids'],np.arange(32))
    from utils.access import read_rows
    y=torch.tensor(read_rows(labels,'actions',np.arange(b)))
    x=torch.tensor(co.history_observations(tr)[:b]);mask=torch.tensor(co.valid_target_mask(tr)[:b]);assert y.shape==(*mask.shape,2)
    assert torch.isfinite(y).all() and y.abs().max()<=1.00001
    # The archived MPE curator already sliced actions at t1..22; verify timestamp contract explicitly.
    assert np.array_equal(co.control_times(tr),np.arange(1,23) if env=='mpe' else np.arange(200))
    val=np.arange(0,b,4);fit=np.setdiff1d(np.arange(b),val);assert len(fit) and len(val)
    torch.manual_seed(seed);net.decoder=ActionDecoder(co.latent_dim,2,hidden=128)
    # Raw standardized history outputs match the existing numerical decoder interface.
    assert not net.latent_mean.any() and torch.equal(net.latent_scale,torch.ones_like(net.latent_scale))
    initial=fingerprint(net.state_dict());history_initial=fingerprint(net.history.state_dict());head_initial=fingerprint(net.decoder.state_dict())
    count=int(mask[fit].sum());batch=min(c['ground_batch'][env],count);updates=c['ground_updates'][env]
    batches=np.random.default_rng(seed+211).integers(count,size=(updates,batch));np.save(dest/'paired_batches.npy',batches)
    save(dest/'label_grant.json',dict(path=str(labels),sha256=sha(labels),budget_episodes=b,target_vectors=int(mask.sum()),fit_vectors=count,validation_vectors=int(mask[val].sum()),partner_vectors=0,local_ids=list(range(b)),control_times=co.control_times(tr).tolist(),archive_contains_32_episodes=True,selected_prefix=b))
    rows=[]
    for route in ('Frozen','Adapt'):
        policy=deepcopy(net);assert fingerprint(policy.state_dict())==initial
        for q in policy.history.parameters():q.requires_grad_(route=='Adapt')
        params=[q for q in policy.parameters() if q.requires_grad];opt=torch.optim.Adam(params,lr=.001);best=float('inf');selected=0;best_state=None
        with torch.no_grad():fixed=co.select_history_outputs(policy.history((x-policy.obs_mean)/policy.obs_scale)[0])
        for step,ix in enumerate(batches,1):
            pred=co.select_history_outputs(policy(x)[0]) if route=='Adapt' else policy.decoder(fixed)
            loss=(pred[fit][mask[fit]][ix]-y[fit][mask[fit]][ix]).square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();assert all(torch.isfinite(q.grad).all() for q in params if q.grad is not None);opt.step()
            if step==1 or step%25==0 or step==updates:
                with torch.no_grad():score=float((co.select_history_outputs(policy(x[val])[0])-y[val]).square()[mask[val]].mean())
                assert np.isfinite(score)
                if score<best:best=score;selected=step;best_state=deepcopy(policy.state_dict())
            if step%100==0 or step==updates:progress('ground_'+route,completed=step,total=updates)
        policy.load_state_dict(best_state);policy.eval()
        with torch.no_grad():
            full=policy(x[:2])[0];hidden=None;pieces=[]
            for t in range(x.shape[1]):
                p,hidden=policy(x[:2,t:t+1],hidden);pieces.append(p)
            error=float((torch.cat(pieces,1)-full).abs().max());assert error<1e-5,error
            poison=x[:2].clone();poison[:,4:]=99;prefix_error=float((policy(poison)[0][:,:4]-full[:,:4]).abs().max());assert prefix_error<1e-5
        changed=fingerprint(policy.history.state_dict())!=history_initial;assert changed==(route=='Adapt')
        torch.save(dict(state_dict=policy.state_dict(),qualification=c['qualification'],route=route,updates=updates,budget=b),dest/f'{route}.pt')
        np.save(dest/f'{route}_predictions.npy',co.select_history_outputs(policy(x)[0]).detach().numpy())
        rows.append(dict(route=route,history_initial=history_initial,decoder_initial=head_initial,inputs=fingerprint({'x':x}),targets=fingerprint({'y':y}),batches=sha(dest/'paired_batches.npy'),normalization=fingerprint({'mean':net.obs_mean,'scale':net.obs_scale}),updates=updates,batch=batch,selected_update=selected,validation_mse=best,history_changed=changed,stream_error=error,prefix_error=prefix_error))
    keys=['history_initial','decoder_initial','inputs','targets','batches','normalization','updates','batch'];assert all(rows[0][k]==rows[1][k] for k in keys)
    return dict(passed=True,qualification=c['qualification'],rows=rows,pairing_checks=keys,simulator_queries=0)

def evaluate(c,env,seed,method,u,dest,progress):
    assert env=='mpe', 'Only the manuscript MPE cohort is exposed'
    import numpy as np
    import torch
    rows=[]
    for route in ('Frozen','Adapt'):
        net=restore(u);net.load_state_dict(torch.load(u/f'ground/{route}.pt',map_location='cpu',weights_only=False)['state_dict']);net.eval()
        seeds=c['evaluation_seeds'][env]
        if env=='mpe':from evaluation.mpe import evaluate as ev;policy=net
        else:
            from evaluation.mamujoco import evaluate as ev
            from training.grounding_mamujoco import GroundedPolicy
            policy=GroundedPolicy(net,np.zeros(105,np.float32),np.ones(105,np.float32))
        episodes=[]
        for i,s in enumerate(seeds):
            if env=='mpe':r=ev(policy,[s])
            else:
                gate=W/c['teacher_gate'];checkpoint=local(read(gate)['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else W/checkpoint
                assert sha(checkpoint)==read(gate)['checkpoint_sha256']
                r=ev(policy,[s],200,teacher_checkpoint=checkpoint,teacher_gate=gate)
            assert len(r['episode_returns'])==1 and np.isfinite(r['episode_returns']).all();episodes.append(r)
            progress('evaluate_'+route,completed=i+1,total=len(seeds))
        values=np.array([r['episode_returns'][0] for r in episodes]);np.save(dest/f'{route}_returns.npy',values)
        row=dict(environment=env,seed=seed,method=method+'-state',route=route,episode_seeds=seeds,episode_returns=values.tolist(),mean_return=float(values.mean()),episodes=episodes,qualification=c['qualification']);save(dest/f'{route}.json',row);rows.append(row)
    return dict(passed=True,qualification=c['qualification'],rows=rows)

def worker(out,stage,env,seed,method):
    assert env=='mpe', 'Only the manuscript MPE cohort is exposed'
    import torch
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    c=read(out/'protocol.json');u=out/identity(env,seed,method);dest=u/stage;dest.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result={};audit=None
    def progress(phase,**kw):save(dest/'progress.json',dict(phase=phase,pid=os.getpid(),active_seconds=time.perf_counter()-start,updated_unix=time.time(),**kw))
    try:
        verify(read(out/'source_manifest.json'))
        if stage!='pre':verify(read(out/('pretrain_freeze.json' if stage=='ground' else 'ground_freeze.json')))
        # Strict artifact allowlist, including model-only constructor assets.
        tr,dv,labels=sources(env,seed);allowed=[tr,dv,*u.rglob('*.pt'),*u.rglob('*.npy'),*u.rglob('*.npz')]
        if stage=='pre':allowed+=list((P/'provenance/flam_perceptual_assets_1c707be_2026_09_21').rglob('*.pth'))
        if stage=='ground':allowed.append(labels)
        if stage=='evaluate':
            gate=W/c['teacher_gate'];cp=local(read(gate)['checkpoint']);allowed.append(cp if cp.is_absolute() else W/cp)
        audit=artifact_guard(dest,allowed,stage)
        result=globals()[stage](c,env,seed,method,u,dest,progress);assert not audit['violations']
        # Manager performs final numeric hash checks outside the stage access boundary.
        save(dest/'status.json',dict(state='complete'))
    except BaseException:
        result.update(passed=False,traceback=traceback.format_exc());save(dest/'status.json',dict(state='failed'));raise
    finally:
        result['active_seconds']=time.perf_counter()-start;save(dest/'result.json',result)
        if audit is not None:save(dest/'access_audit.json',audit)
