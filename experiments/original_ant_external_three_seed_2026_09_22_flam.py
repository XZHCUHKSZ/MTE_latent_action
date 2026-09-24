"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

from dataclasses import asdict

import argparse

import gc

import json

import os

from pathlib import Path

import random

import shutil

import sys

import time

import traceback

from experiments.external_flam_config import author_config,sha

P=Path(__file__).resolve().parents[1];W=P.parents[1]

P=Path(__file__).resolve().parents[1];W=P.parents[1]

OUT=None

CONFIG=P/'configs/original_ant_external_three_seed_2026_09_22.json'

VENDOR=P/'third_party/flam_source_complete_1c707be'

REPORT=P/'results/baseline_visual_slot_research_2026_09_21'

ASSETS=P/'provenance/flam_perceptual_assets_1c707be_2026_09_21'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def save(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=True,indent=2,default=str),encoding='utf-8')

def main():
    assert sys.platform=='linux'
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);a=parser.parse_args()
    protocol=read(CONFIG);assert protocol['launch_locked'] and a.seed in protocol['seeds']
    c=dict(protocol['upstream']['flam'],seed=a.seed)
    assert not c['qualification'] and not c['performance'] and not c['cost_only']
    assert (c['tokenizer_updates'],c['factored_updates'],c['tokenizer_batch'],c['factored_batch'])==(5000,1920,64,32)
    assert read(REPORT/'flam_author_batch_independent_audit_2026_09_21.json')['passed']
    OUT=P/'outputs/original_ant_external_three_seed_2026_09_22'/f'seed{a.seed}'/'flam'
    OUT.mkdir(exist_ok=False);start=time.perf_counter();result={};denied=[];reads=[]
    try:
        header=Path(read(REPORT/'flam_wsl_headers_install_r2.json')['header']);assert header.exists()
        os.environ['C_INCLUDE_PATH']=str(header.parent)+':'+str(header.parent.parent)
        files=[W/f for f in c['train_files']];assert len(files)==32 and len(set(files))==32
        manifest={str(VENDOR/k):h for k,h in read(REPORT/'flam_complete_source_manifest.json')['files'].items()}
        other=[P/'utils/linux_process_identity.py',REPORT/'flam_author_batch_preflight_2026_09_21.json',Path(__file__).with_name('original_ant_flam_multitrajectory_cost_2026_09_21.py'),Path(__file__),CONFIG,Path(__file__).with_name('original_ant_flam_author_qualification_r2_2026_09_21.py'),
               ASSETS/'lpips.pth',ASSETS/'torch_hub/checkpoints/vgg16-397923af.pth',*files]
        manifest.update({str(f):sha(f) for f in other});assert all(sha(f)==h for f,h in manifest.items())
        save('source_and_input_manifest.json',manifest);save('protocol.json',c)
        def guard(event,args):
            if event=='import' and str(args[0]).split('.')[0] in {'mujoco','mujoco_py','pettingzoo','mpe2'}:
                denied.append(str(args[0]));raise PermissionError('Simulator forbidden')
            if event=='socket.connect':raise PermissionError('Network forbidden during training')
            if event=='open' and isinstance(args[0],(str,bytes)):
                path=Path(os.fsdecode(args[0])).resolve();s=str(path)
                if any(part in s for part in ['/labels/','/target_labels/','/curator/','/rgb/dev/','/rgb/eval/']):
                    denied.append(s);raise PermissionError('Native labels or held-out data forbidden')
                if '/rgb/train/' in s:
                    assert path in files;reads.append(str(path))
        sys.addaudithook(guard)
        sys.path[:0]=[str(VENDOR),str(VENDOR/'third_party_models/cosmos')]
        import numpy as np
        import torch
        import gymnasium
        import importlib.util
        identity_spec=importlib.util.spec_from_file_location('flam_cost_process_identity',P/'utils/linux_process_identity.py')
        identity_module=importlib.util.module_from_spec(identity_spec);identity_spec.loader.exec_module(identity_module)
        from models.tokenizer.tokenizer import Tokenizer
        from models.lam.lam_factored import FactoredLatentActionModel
        from data.data_utils.dataclass import SubTrajectory
        from training.trainer import Trainer
        import models.modules.loss.lpips as LP
        def deny_env(*args,**kwargs):denied.append('gymnasium.make');raise PermissionError('No environment construction')
        gymnasium.make=deny_env
        for action in [lambda:open(W/'qualification_negative/labels/test.npy','rb'),lambda:__import__('mujoco'),lambda:gymnasium.make('Ant-v5')]:
            try:action()
            except PermissionError:pass
            else:raise AssertionError('Access boundary failed')
        proc=identity_module.LinuxProcessIdentity();torch.set_num_threads(2)
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        torch.hub.set_dir(str(ASSETS/'torch_hub'));LP.REPO_PATH=OUT/'perceptual_path_adapter'
        lp=LP.REPO_PATH/'models/modules/loss/lpips.pth';lp.parent.mkdir(parents=True);shutil.copyfile(ASSETS/'lpips.pth',lp)
        assert sha(lp)==sha(ASSETS/'lpips.pth')
        cfg=author_config(c);save('effective_author_config.json',asdict(cfg))
        def status(stage,**kw):save('status.json',dict(state='running',stage=stage,pid=proc.pid,created=proc.create_time(),updated_unix=time.time(),active_seconds=time.perf_counter()-start,**kw))
        status('load_training_observations');load_start=time.perf_counter();data=[]
        for f in files:
            rgb=np.load(f,allow_pickle=False);assert rgb.shape==(201,4,64,64,3) and rgb.dtype==np.uint8
            tiled=np.concatenate([np.concatenate([rgb[:,0],rgb[:,1]],axis=2),np.concatenate([rgb[:,2],rgb[:,3]],axis=2)],axis=1)
            for i,(y,x) in enumerate([(0,0),(0,64),(64,0),(64,64)]):assert np.array_equal(tiled[:,y:y+64,x:x+64],rgb[:,i])
            data.append(tiled.transpose(0,3,1,2).copy())
        load_seconds=time.perf_counter()-load_start;rng=np.random.default_rng(c['seed']);plans={}
        for stage,length in [('tokenizer',1),('factored',10)]:
            population=np.array([(e,j) for e in range(32) for j in range(202-length)],dtype=np.int64)
            batch_size=c[stage+'_batch'];usable=len(population)//batch_size*batch_size
            plans[stage]=np.concatenate([population[rng.permutation(len(population))[:usable]] for _ in range(c[stage+'_epochs'])]).reshape(c[stage+'_updates'],batch_size,2).tolist()
        save('sample_plan.json',plans)
        def train(model,stage,length,clip):
            model.train();opt=model.configure_optimizers();scaler=torch.amp.GradScaler()
            sched=Trainer.create_warmup_cosine_scheduler(None,opt,c[stage+'_updates'])
            rows=[];torch.cuda.reset_peak_memory_stats();stage_start=time.perf_counter()
            for step,indices in enumerate(plans[stage]):
                if step%10==0:status(stage,completed=step,total=len(plans[stage]))
                torch.cuda.synchronize();tick=time.perf_counter()
                image=torch.from_numpy(np.stack([data[e][j:j+length] for e,j in indices])).cuda().float()/255
                batch=SubTrajectory(image=image);assert batch.action is None and batch.reward is None
                model.zero_grad()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,logs=model(batch,is_last_batch=False)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(opt)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
                assert torch.isfinite(grad) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                scaler.step(opt);scaler.update();sched.step()
                if hasattr(model,'update_annealing'):model.update_annealing((step+1)/len(plans[stage]))
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                rows.append(dict(update=step+1,seconds=elapsed,loss=float(loss.detach()),grad_norm=float(grad),last_batch_diagnostics=False))
                if (step+1)%10==0:save(stage+'_progress.json',rows)
            measured=np.array([r['seconds'] for r in rows[3:-1]])
            return dict(updates=len(rows),rows=rows,stage_seconds=time.perf_counter()-stage_start,
                        measurement='End-to-end update includes CPU sample assembly, host-to-GPU copy, forward/backward/optimizer; steady range updates4 through penultimate; training image logging disabled',
                        steady_mean_seconds=float(measured.mean()),steady_p50_seconds=float(np.median(measured)),steady_p90_seconds=float(np.quantile(measured,.9)),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        sampled_training_episodes=sorted({e for batch in plans[stage] for e,j in batch}))
        status('tokenizer_initialization');tokenizer=Tokenizer(cfg).cuda()
        tok_initial={n:p.detach().clone() for n,p in tokenizer.encoder.named_parameters()}
        tok=train(tokenizer,'tokenizer',1,cfg.tokenizer.grad_norm_clip)
        assert any(not torch.equal(v,tok_initial[n]) for n,v in tokenizer.encoder.named_parameters());del tok_initial
        tokenizer.eval();torch.save(dict(state_dict=tokenizer.state_dict(),updates=c['tokenizer_updates'],qualification=False,cost_only=False),OUT/'tokenizer.pt')
        frozen_hash=sha(OUT/'tokenizer.pt');save('tokenizer_freeze.json',dict(sha256=frozen_hash,updates=c['tokenizer_updates']))
        tok_inventory=dict(total=sum(p.numel() for p in tokenizer.parameters()),trainable=sum(p.numel() for p in tokenizer.parameters() if p.requires_grad))
        assert tok_inventory==dict(total=28018272,trainable=13302112)
        status('factored_initialization');model=FactoredLatentActionModel(cfg,tokenizer=tokenizer).cuda()
        frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n.startswith(('encoder.','decoder.','tokenizer_quantizer.'))}
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        del tokenizer;gc.collect();torch.cuda.empty_cache()
        assert sum(p.numel() for p in model.parameters())==38663008
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==10644736
        fact=train(model,'factored',10,cfg.lam.grad_norm_clip)
        assert any(not torch.equal(v.detach().cpu(),initial[n]) for n,v in model.named_parameters() if v.requires_grad)
        assert all(torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in frozen.items())
        torch.save(dict(state_dict=model.state_dict(),updates=c['factored_updates'],qualification=False,cost_only=False,config=asdict(cfg)),OUT/'factored.pt')
        assert sha(OUT/'tokenizer.pt')==frozen_hash and all(sha(f)==h for f,h in manifest.items()) and len(denied)==3
        assert set(reads)==set(map(str,files))
        result=dict(passed=True,qualification=False,cost_only=False,performance=False,tokenizer=tok,factored=fact,
                    loading_seconds=load_seconds,cpu_rgb_bytes=sum(x.nbytes for x in data),source_files=len(manifest),
                    source_hashes_unchanged=True,training_episode_files=32,training_frames=6432,lossless_camera_tiling=True,
                    native_action_arrays_read=0,simulator_queries=0,negative_access_checks=denied,frozen_tokenizer_arrays=len(frozen),
                    tokenizer_frozen_during_factored=True,gpu=torch.cuda.get_device_name(),
                    parameter_inventory=dict(tokenizer=tok_inventory,factored=dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))),
                    limitations=['Full5000 tokenizer/1920 factored updates at batches64/32; downstream control evaluation remains separate','No full-batch image logging during training; zero-weight factored image diagnostics skipped, tokenizer LPIPS retained','No dev/eval data, native actions, control returns or reward-based model selection in upstream training'])
        save('status.json',dict(state='complete',pid=proc.pid,created=proc.create_time()))
    except BaseException:
        result.update(passed=False,qualification=False,cost_only=False,traceback=traceback.format_exc());save('status.json',dict(state='failed',pid=os.getpid()));raise
    finally:
        result['active_seconds']=time.perf_counter()-start;save('result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['tokenizer','factored']},ensure_ascii=True,default=str))

if __name__=='__main__':main()
