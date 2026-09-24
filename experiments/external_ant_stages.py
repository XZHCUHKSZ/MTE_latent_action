"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

from copy import deepcopy

import hashlib

import json

import os

from pathlib import Path

import sys

import time

import traceback

P = Path(__file__).resolve().parents[1]

W = P.parents[1]

NAME = Path(__file__).stem

CONFIG = P/'configs/original_ant_external_three_seed_downstream_2026_09_22_r2.json'

OUT = P/'outputs'/NAME

UP = P/'outputs/original_ant_external_three_seed_2026_09_22'

ARCHIVE = W/'visual_multiagent_2026_09_10/seed5_v1/runtime'

RGB = W/'visual_multiagent_2026_09_10/pixel_pipeline_v1/data/pilot/rgb'

LABELS = W/'visual_multiagent_2026_09_10/pixel_pipeline_v1/runtime/pilot/validate/target_labels/train'

ASSETS = P/'outputs/visual_supervision_inventory_2026_09_19/assets'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p, v):
    p = Path(p); t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v, ensure_ascii=True, indent=2), encoding='utf-8'); t.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def local(p):
    s = str(p).replace('\\','/')
    if sys.platform=='linux' and s[1:3]==':/': return Path('/mnt/'+s[0].lower()+'/'+s[3:])
    if sys.platform=='win32' and s.startswith('/mnt/d/'): return Path('D:/'+s[7:])
    return Path(s)

def verify(m):
    for p,h in m.items(): assert sha(local(p))==h, p

def unit(seed, method): return OUT/f'seed{seed}_{method}'

def obs_files(): return [RGB/s/f'{i:04d}.npy' for s,n in [('train',32),('dev',8)] for i in range(n)]

def access_guard(stage):
    report = dict(stage=stage, label_reads=[], denied=[])
    labels = {p.resolve() for p in [LABELS/f'{i:04d}.npy' for i in range(2)]}
    def guard(event,args):
        if stage!='evaluate' and event=='import' and str(args[0]).split('.')[0] in {'mujoco','mujoco_py','pettingzoo','mpe2'}:
            report['denied'].append(str(args[0])); raise PermissionError('Simulator forbidden')
        if event=='socket.connect': raise PermissionError('Network forbidden')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve(); s=str(p).replace('\\','/').lower()
            if any(k in s for k in ['/labels/','/target_labels/','/curator/','/rgb/eval/']):
                if stage=='ground' and p in labels: report['label_reads'].append(str(p))
                elif stage=='evaluate' and p in {(ASSETS/f'labels/dev/{i:04d}.npy').resolve() for i in range(8)}:
                    report['label_reads'].append(str(p))
                else: report['denied'].append(str(p)); raise PermissionError('Unapproved label or test-data access')
    sys.addaudithook(guard)
    if stage in ('export','history'):
        for fn in [lambda:open(LABELS/'0000.npy','rb'),lambda:__import__('mujoco')]:
            try: fn()
            except PermissionError: pass
            else: raise AssertionError('Negative access check failed')
    return report

def export(seed, method, dest, progress):
    import numpy as np
    import torch
    u=UP/f'seed{seed}'/method; r=read(u/'result.json')
    assert r['passed'] and not r['qualification'] and not r['cost_only']
    if method=='otf':
        from omegaconf import OmegaConf
        sys.path.insert(0,str(P/'third_party/otf_lam_official'))
        from otf_vqvae.model import OTFVQVAE
        from otf_lam_pixels.model import FrozenOTFVQVAEPatchExtractor, OTFLAMPixels
        v=torch.load(u/'vq.pt',map_location='cuda',weights_only=False)
        m=torch.load(u/'pixel.pt',map_location='cuda',weights_only=False)
        assert v['updates']==2000 and m['updates']==20000 and not v['qualification'] and not m['qualification']
        vq=OTFVQVAE(OmegaConf.create(v['config'])).cuda(); vq.load_state_dict(v['state_dict'],strict=True)
        model=OTFLAMPixels(FrozenOTFVQVAEPatchExtractor(vq),OmegaConf.create(m['config'])).cuda()
        model.load_state_dict(m['state_dict'],strict=True); del m,v
    else:
        assert sys.platform=='linux'
        import shutil
        sys.path.insert(0,str(P/'experiments'))
        from experiments.external_flam_config import author_config
        report=P/'results/baseline_visual_slot_research_2026_09_21'
        header=local(read(report/'flam_wsl_headers_install_r2.json')['header'])
        os.environ['C_INCLUDE_PATH']=str(header.parent)+':'+str(header.parent.parent)
        vendor=P/'third_party/flam_source_complete_1c707be'
        sys.path[:0]=[str(vendor),str(vendor/'third_party_models/cosmos')]
        from models.tokenizer.tokenizer import Tokenizer
        from models.lam.lam_factored import FactoredLatentActionModel
        from data.data_utils.dataclass import SubTrajectory
        import models.modules.loss.lpips as LP
        import gymnasium
        gymnasium.make=lambda *a,**kw: (_ for _ in ()).throw(PermissionError('No environment during export'))
        assets=P/'provenance/flam_perceptual_assets_1c707be_2026_09_21'
        torch.hub.set_dir(str(assets/'torch_hub')); LP.REPO_PATH=dest/'perceptual_path_adapter'
        lp=LP.REPO_PATH/'models/modules/loss/lpips.pth'; lp.parent.mkdir(parents=True)
        shutil.copyfile(assets/'lpips.pth',lp)
        cfg=author_config(dict(read(UP/'protocol.json')['upstream']['flam'],seed=seed))
        tok=Tokenizer(cfg).cuda(); tc=torch.load(u/'tokenizer.pt',map_location='cuda',weights_only=False)
        assert tc['updates']==5000 and not tc['qualification']; tok.load_state_dict(tc['state_dict'],strict=True)
        model=FactoredLatentActionModel(cfg,tokenizer=tok).cuda()
        ck=torch.load(u/'factored.pt',map_location='cuda',weights_only=False)
        assert ck['updates']==1920 and not ck['qualification']; model.load_state_dict(ck['state_dict'],strict=True)
        del tok,tc,ck
    model.eval()
    for q in model.parameters(): q.requires_grad_(False)
    outputs=[]; batch_parity=[]
    with torch.no_grad():
        for e,f in enumerate(obs_files()):
            rgb=np.load(f,allow_pickle=False); assert rgb.shape==(201,4,64,64,3) and rgb.dtype==np.uint8
            mosaic=np.concatenate([np.concatenate([rgb[:,0],rgb[:,1]],2),np.concatenate([rgb[:,2],rgb[:,3]],2)],1)
            frames=torch.from_numpy(mosaic.transpose(0,3,1,2).copy()).cuda().float()/255
            if method=='otf':
                def encode(ts):
                    return model.action_labels(dict(current=frames[ts],next=frames[ts+1],reference_frame=frames[ts]))['z_act'].float()
                parts=[encode(torch.arange(lo,min(lo+16,200),device='cuda')) for lo in range(0,200,16)]
                z=torch.cat(parts)
                error=float((z[:1]-encode(torch.tensor([0],device='cuda'))).abs().max())
            else:
                def encode(ts):
                    # Same-length rolling contexts; no padding or cross-episode mixing.
                    image=torch.stack([frames[max(0,t-8):t+2] for t in ts])
                    batch=SubTrajectory(image=image); assert batch.action is None and batch.reward is None
                    with torch.autocast('cuda',dtype=torch.bfloat16): zz=model.encode(batch)[3]
                    assert zz.shape[2:]==(1,4,256)
                    return zz[:,-1,0].reshape(len(ts),1024).float()
                parts=[encode([t]) for t in range(200)]
                z=torch.cat(parts)
                error=float((z[9:10]-encode([9])).abs().max())
            assert torch.isfinite(z).all() and error<.001,(method,error)
            batch_parity.append(error); outputs.append(z.cpu().numpy())
            progress('export',completed=e+1,total=40)
    z=np.stack(outputs); assert z.shape==(40,200,256 if method=='otf' else 1024)
    np.save(dest/'latents.npy',z)
    return dict(passed=True,qualification=False,formal=False,shape=list(z.shape),updates=0,action_arrays_read=0,
                simulator_queries=0,batch_single_max_errors=batch_parity,all_slots_retained=method=='flam',
                context='OTF adjacent pair; FLAM rolling maximum10 frames, four slots concatenated')

def history(seed, method, dest, progress):
    import numpy as np
    from training.representation_mamujoco import train_history_policy
    x=np.load(ARCHIVE/str(seed)/'features/features.npz')['x'][:,:,0]
    z=np.load(unit(seed,method)/'export/latents.npy')
    assert x.shape==(40,201,32)
    r=train_history_policy(x,z,np.ones((40,200),bool),32,seed,256,dest/'policy',progress,method='edge_cara')
    return dict(passed=True,qualification=False,formal=False,history=r,action_arrays_read=0,simulator_queries=0)

def ground(seed, method, dest, progress):
    import numpy as np
    import torch
    from closed_loop_lam_v1 import common as C
    from training.visual_supervision_control import fingerprint
    u=unit(seed,method); ck=torch.load(u/'history/policy/policy.pt',map_location='cuda',weights_only=False)
    dim=ck['z_dim']; assert ck['native_action_labels_read']==0 and ck['frozen_before_grounding']
    net=C.RecurrentPolicy(32,dim,hidden=128).cuda().eval(); net.load_state_dict(ck['state_dict'],strict=True)
    x=np.load(ARCHIVE/str(seed)/'features/features.npz')['x'][:2,:200,0]
    x=torch.tensor((x-ck['obs_mean'])/ck['obs_std'],device='cuda')
    with torch.no_grad():
        z=net(x)[0].reshape(400,dim); mean=z.mean(0); scale=z.std(0).clamp_min(.0001); fixed=(z-mean)/scale
    C.seed_all(seed); head=C.ActionDecoder(dim,2,hidden=128).cuda()
    batches=np.random.default_rng(seed).integers(400,size=(600,256)); np.save(dest/'paired_batches.npy',batches)
    initial_history=fingerprint(net.state_dict()); initial_decoder=fingerprint(head.state_dict())
    files=[LABELS/f'{i:04d}.npy' for i in range(2)]
    save(dest/'label_grant.json',dict(files={str(f):sha(f) for f in files},target_vectors=400,partner_vectors=0,fit=400,action_validation=0))
    y=torch.tensor(np.stack([np.load(f) for f in files]),device='cuda').reshape(400,2); rows=[]
    for arm in ('Frozen','Adapt'):
        folder=u/('ground_'+arm); folder.mkdir(exist_ok=False)
        policy=deepcopy(net); decoder=deepcopy(head)
        for q in policy.parameters(): q.requires_grad_(arm=='Adapt')
        policy.train(arm=='Adapt'); decoder.train()
        params=list(decoder.parameters())+(list(policy.parameters()) if arm=='Adapt' else [])
        opt=torch.optim.Adam(params,lr=.001); curve=[]
        for step,ix in enumerate(batches):
            values=(policy(x)[0].reshape(400,dim)-mean)/scale if arm=='Adapt' else fixed
            loss=(decoder(values[ix])-y[ix]).square().mean(); assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); assert all(torch.isfinite(q.grad).all() for q in params if q.grad is not None)
            opt.step()
            if step%100==0 or step==599:
                curve.append(dict(update=step+1,loss=float(loss.detach()))); progress('ground',arm=arm,completed=step+1,total=600)
        policy.eval(); decoder.eval()
        with torch.no_grad():
            pred=decoder((policy(x)[0]-mean)/scale); hidden=None; parts=[]
            for t in range(200):
                zz,hidden=policy(x[:,t:t+1],hidden); parts.append(decoder((zz-mean)/scale))
            err=float((pred-torch.cat(parts,1)).abs().max()); assert err<1e-5,err
            poison=x.clone(); poison[:,100:]=99
            prefix_error=float((decoder((policy(poison)[0]-mean)/scale)[:,:100]-pred[:,:100]).abs().max()); assert prefix_error<1e-5
        changed=fingerprint(policy.state_dict())!=initial_history; assert changed==(arm=='Adapt')
        torch.save(dict(history=policy.state_dict(),decoder=decoder.state_dict(),obs_mean=ck['obs_mean'],obs_std=ck['obs_std'],
                        mean=mean,scale=scale,z_dim=dim,seed=seed,method=method,arm=arm,qualification=False,formal=False,
                        updates=600,budget=2),folder/'controller.pt')
        np.save(folder/'fit_predictions.npy',pred.detach().cpu().numpy())
        row=dict(passed=True,arm=arm,updates=600,indices_per_update=256,history_initial=initial_history,decoder_initial=initial_decoder,
                 normalization=fingerprint(dict(mean=mean,scale=scale,obs_mean=ck['obs_mean'],obs_std=ck['obs_std'])),
                 inputs=fingerprint(dict(x=x)),targets=fingerprint(dict(y=y)),batches=sha(dest/'paired_batches.npy'),
                 history_parameters=sum(q.numel() for q in policy.parameters()),decoder_parameters=sum(q.numel() for q in decoder.parameters()),
                 history_changed=changed,stream_error=err,prefix_error=prefix_error,fit_mse=float((pred.reshape(400,2)-y).square().mean()),curve=curve)
        save(folder/'result.json',row); rows.append(row)
    keys=['history_initial','decoder_initial','normalization','inputs','targets','batches','history_parameters','decoder_parameters']
    assert all(rows[0][k]==rows[1][k] for k in keys)
    return dict(passed=True,qualification=False,formal=False,arms=rows,pairing_checks=keys,action_vectors=400,simulator_queries=0)

def evaluate(seed, method, dest, progress):
    import numpy as np
    import torch
    from closed_loop_lam_v1 import common as C
    from visual import train as V, rollout as R, render as RR, dcs_source as D, scene as S
    c=read(CONFIG); u=unit(seed,method); V.SEED=seed; V.RT=ARCHIVE/str(seed); V.OLD=ASSETS
    RR.ROOT=W; D.IMAGES=S.IMAGES=W/'visual_multiagent_2026_09_10/laom_complexity_v1/assets/DAVIS/JPEGImages/480p'
    R.RT=u; R.ARMS=['Frozen','Adapt']; R.SEEDS=c['evaluation_seeds']; R.progress=progress
    def controller(arm):
        ck=torch.load(u/('ground_'+arm)/'controller.pt',map_location='cuda',weights_only=False)
        assert not ck['qualification'] and ck['updates']==600
        net=C.RecurrentPolicy(32,ck['z_dim'],hidden=128).cuda().eval(); net.load_state_dict(ck['history'],strict=True)
        head=C.ActionDecoder(ck['z_dim'],2,hidden=128).cuda().eval(); head.load_state_dict(ck['decoder'],strict=True)
        def predict(ff,h=None):
            z,h=net((ff-torch.as_tensor(ck['obs_mean'],device='cuda'))/torch.as_tensor(ck['obs_std'],device='cuda'),h)
            return head((z-ck['mean'])/ck['scale']),h
        return predict
    R.controller=controller; rows=[]
    for arm in R.ARMS:
        folder=dest/arm; folder.mkdir(exist_ok=False)
        r=R.evaluate(folder,arm); assert len(r['rows'])==len(c['evaluation_seeds'])
        r.update(seed=seed,method=method,qualification=False,formal=False,evidence_class='fixed existing-seed development pilot')
        save(folder/'result.json',r); rows.append(r)
    return dict(passed=True,qualification=False,formal=False,rows=rows)

def worker(stage,seed,method):
    dest=unit(seed,method)/stage; dest.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    def progress(phase,**kw): save(dest/'progress.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**kw))
    result={}; report=None
    try:
        verify(read(OUT/'source_manifest.json'))
        if stage in ('ground','evaluate'): verify(read(OUT/('pretrain_freeze.json' if stage=='ground' else 'ground_freeze.json')))
        report=access_guard(stage)
        import torch
        torch.set_num_threads(2); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        if stage!='export':
            from closed_loop_lam_v1 import common as C
            C.seed_all(seed); torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        result=globals()[stage](seed,method,dest,progress)
        verify(read(OUT/'source_manifest.json'))
        if stage in ('ground','evaluate'): verify(read(OUT/('pretrain_freeze.json' if stage=='ground' else 'ground_freeze.json')))
        save(dest/'status.json',dict(state='complete'))
    except BaseException:
        result.update(passed=False,traceback=traceback.format_exc()); save(dest/'status.json',dict(state='failed')); raise
    finally:
        result['active_seconds']=time.perf_counter()-start; save(dest/'result.json',result)
        if report is not None: save(dest/'access_audit.json',report)
