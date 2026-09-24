"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import argparse

import ast

import hashlib

import json

import os

from pathlib import Path

import sys

import time

import traceback

P = Path(__file__).resolve().parents[1]

W = P.parents[1]

V = P / 'third_party/otf_lam_official'

C = P / 'configs/original_ant_external_three_seed_2026_09_22.json'

OUT = None

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);a=parser.parse_args()
    protocol=json.loads(C.read_text(encoding='utf-8'));assert protocol['launch_locked'] and a.seed in protocol['seeds']
    c=dict(protocol['upstream']['otf'],seed=a.seed)
    OUT=P/'outputs/original_ant_external_three_seed_2026_09_22'/f'seed{a.seed}'/'otf'
    assert not c['qualification'] and not c['cost_only'] and not c['performance']
    assert (c['vq_updates'], c['pixel_updates'], c['vq_warmup_updates']) == (2000, 20000, 200)
    assert (c['vq_batch'], c['pixel_batch'], c['transition_offset']) == (32, 8, 1)
    assert not OUT.exists(), 'Existing run must never be retried or overwritten'
    import numpy as np
    import torch
    import psutil
    from omegaconf import DictConfig, OmegaConf
    assert torch.cuda.is_available()
    def ancestor_ids(proc):
        ids=set()
        while proc is not None:
            try: parent=proc.parent()
            except psutil.NoSuchProcess: break
            if parent is None or parent.pid in ids: break
            ids.add(parent.pid); proc=parent
        return ids
    ancestors=ancestor_ids(psutil.Process())
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.pid == os.getpid():
            continue
        cmd = proc.info['cmdline'] or []
        # Exact module/script identity, not a shell command containing its name.
        if any(Path(x).name == Path(__file__).name or x == 'experiments.' + Path(__file__).stem for x in cmd[1:]):
            # Windows venv launcher is an ancestor, not a second manager.
            assert proc.pid in ancestors, cmd
    old = json.loads((P / 'outputs/original_ant_otf_deployment_qualification_2026_09_21/upstream/source_manifest.json').read_text())
    vendor = sorted(V.rglob('*.py')) + sorted(V.rglob('*.yaml'))
    for path in vendor:
        assert str(path) in old and sha(path) == old[str(path)], path
    files = [W / c['data'] / f'{i:04d}.npy' for i in range(32)]
    for path in files:
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        assert arr.shape == (201, 4, 64, 64, 3) and arr.dtype == np.uint8
    OUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(passed=False, qualification=False, cost_only=False, performance=False, stage='full_upstream')
    proc = psutil.Process()
    identity = dict(pid=proc.pid, created=proc.create_time(), command=proc.cmdline())
    def status(stage, **kw):
        save('status.json', dict(state='running', stage=stage, **identity,
             updated_unix=time.time(), active_seconds=time.perf_counter()-start, **kw))
    try:
        manifest = {str(p): sha(p) for p in [*vendor, Path(__file__).resolve(), C, *files]}
        save('source_and_input_manifest.json', manifest)
        save('protocol.json', c)
        denied, reads = [], []
        allowed = {p.resolve() for p in files}
        def guard(event, args):
            if event == 'import' and str(args[0]).split('.')[0] in {'mujoco', 'gym', 'gymnasium', 'pettingzoo', 'mpe2'}:
                denied.append(str(args[0])); raise PermissionError('Simulator forbidden')
            if event == 'socket.connect':
                raise PermissionError('Network forbidden')
            if event == 'open' and isinstance(args[0], (str, bytes)):
                path = Path(os.fsdecode(args[0])).resolve()
                s = str(path).replace('\\', '/').lower()
                if any(k in s for k in ('/curator/', '/labels/', '/target_labels/', '/rgb/dev/', '/rgb/eval/')):
                    denied.append(s); raise PermissionError('Labels and held-out observations forbidden')
                if '/rgb/train/' in s:
                    assert path in allowed, path
                    reads.append(str(path))
        sys.addaudithook(guard)
        for action in (lambda: open(P/'labels/negative.npy', 'rb'),
                       lambda: open(files[0].parent.parent/'dev/0000.npy', 'rb'),
                       lambda: __import__('gymnasium')):
            try:
                action()
            except PermissionError:
                pass
            else:
                raise AssertionError('Negative access check failed')
        torch.set_num_threads(2)
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, str(V))
        from otf_vqvae.model import OTFVQVAE, initialize_codebook_with_kmeans, make_motion_signal
        from otf_lam_pixels.model import FrozenOTFVQVAEPatchExtractor, OTFLAMPixels
        cfg = OmegaConf.load(V/'configs/otf_vqvae/walker.yaml')
        cfg.seed = c['seed']
        cfg.data.image_height = cfg.data.image_width = c['image_size']
        cfg.model.patch_height = cfg.model.patch_width = c['patch_size']
        vc = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        tree = ast.parse((V/'otf_lam_pixels/train.py').read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'str2bool', 'parse_args', 'make_config'}]
        assert len(nodes) == 3
        ns = dict(argparse=argparse, Path=Path, DictConfig=DictConfig, OmegaConf=OmegaConf)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(V/'otf_lam_pixels/train.py'), 'exec'), ns)
        argv = sys.argv[:]
        try:
            sys.argv = ['cost', '--otf_vqvae_checkpoint_path', str(OUT/'vq.pt'), '--data_dir', str(files[0].parent),
                        '--output_dir', str(OUT), '--artifact_root', str(OUT), '--run_name', OUT.name,
                        '--use_wandb', 'false', '--batch_size', str(c['pixel_batch']),
                        '--max_steps', str(c['pixel_updates']), '--seed', str(c['seed'])]
            pc = ns['make_config'](ns['parse_args'](), vc, OUT)
        finally:
            sys.argv = argv
        save('effective_configs.json', dict(vq=OmegaConf.to_container(vc, resolve=True), pixel=OmegaConf.to_container(pc, resolve=True)))
        status('load_training_rgb'); load_start = time.perf_counter(); data = []
        for path in files:
            rgb = np.load(path, allow_pickle=False)
            tiled = np.concatenate([np.concatenate([rgb[:,0], rgb[:,1]], axis=2),
                                    np.concatenate([rgb[:,2], rgb[:,3]], axis=2)], axis=1)
            for i, (y,x) in enumerate([(0,0),(0,64),(64,0),(64,64)]):
                assert np.array_equal(tiled[:,y:y+64,x:x+64], rgb[:,i])
            data.append(tiled.transpose(0,3,1,2).copy())
        loading = time.perf_counter()-load_start
        rng = np.random.default_rng(c['seed']); plans = {}
        for stage, updates, batch_size in [('vq',c['vq_updates'],32), ('kmeans',int(vc.kmeans_batches),32), ('pixel',c['pixel_updates'],8)]:
            count = updates*batch_size
            episodes = np.tile(np.arange(32), (count+31)//32)[:count]; rng.shuffle(episodes)
            times = rng.integers(0,200,size=count)
            plans[stage] = np.stack([episodes,times],axis=-1).reshape(updates,batch_size,2).tolist()
        save('sample_plan.json', plans)
        def batch(indices):
            current = torch.from_numpy(np.stack([data[e][t] for e,t in indices])).cuda().float()/255
            nxt = torch.from_numpy(np.stack([data[e][t+1] for e,t in indices])).cuda().float()/255
            return dict(current=current, next=nxt, reference_frame=current)
        status('vq_initialization')
        vq = OTFVQVAE(vc).cuda().train()
        opt = torch.optim.AdamW(vq.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
        counts = dict(vq=sum(p.numel() for p in vq.parameters()))
        assert counts['vq'] == 300483
        initial = {n:p.detach().cpu().clone() for n,p in vq.named_parameters()}
        stages = {}; kmeans_seconds = None; model = None
        for stage in ('vq','pixel'):
            rows = []; torch.cuda.reset_peak_memory_stats(); tick_stage=time.perf_counter()
            if stage == 'pixel':
                frozen = {n:v.detach().cpu().clone() for n,v in vq.state_dict().items()}
                torch.save(dict(state_dict=frozen, config=OmegaConf.to_container(vc,resolve=True), qualification=False, updates=c['vq_updates']), OUT/'vq.pt')
                model = OTFLAMPixels(FrozenOTFVQVAEPatchExtractor(vq), pc).cuda().train()
                parameters = list(model.trainable_parameters()); counts['pixel_trainable'] = sum(p.numel() for p in parameters)
                assert counts['pixel_trainable'] == 8059043
                opt = torch.optim.AdamW(parameters, lr=float(pc.train.lr), weight_decay=float(pc.train.weight_decay))
                pixel_initial = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            for index, ids in enumerate(plans[stage]):
                step=index+1
                if stage=='vq' and step==c['vq_warmup_updates']+1:
                    status('kmeans_initialization'); torch.cuda.synchronize(); kstart=time.perf_counter()
                    initialize_codebook_with_kmeans(vq, [batch(ix) for ix in plans['kmeans']], torch.device('cuda'), vc, step)
                    torch.cuda.synchronize(); kmeans_seconds=time.perf_counter()-kstart; vq.train()
                    assert vq.quantizer.is_initialized()
                torch.cuda.synchronize(); tick=time.perf_counter(); b=batch(ids)
                if stage=='vq':
                    motion=make_motion_signal(b,vq.motion_input_type,vq.motion_transform)
                    o=vq(motion,reference_frame=b['reference_frame'],use_quantization=step>c['vq_warmup_updates'])
                    loss=(o['reconstruction']-motion).square().mean()+float(cfg.loss.lambda_code)*o['code_loss']+float(cfg.loss.lambda_commit)*o['commit_loss']+float(cfg.loss.lambda_orth)*o['orth_loss']
                    parameters=list(vq.parameters())
                else:
                    o=model(b); loss=(o['pred']-o['target']).square().mean()
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True); loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
                assert torch.isfinite(grad)
                assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
                opt.step()
                if stage=='vq' and step>c['vq_warmup_updates']:
                    vq.quantizer.update_codebook(o['patch_embeddings'],o['indices'],step)
                torch.cuda.synchronize()
                rows.append(dict(update=step,seconds=time.perf_counter()-tick,loss=float(loss.detach()),grad_norm=float(grad)))
                if step%10==0:
                    save(stage+'_progress.json',rows); status(stage,completed=step,total=c[stage+'_updates'])
            values=np.array([r['seconds'] for r in rows[20:]])
            stages[stage]=dict(rows=rows,stage_seconds=time.perf_counter()-tick_stage,
                steady_mean_seconds=float(values.mean()),steady_p90_seconds=float(np.quantile(values,.9)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if stage=='vq':
                assert any(not torch.equal(initial[n],p.detach().cpu()) for n,p in vq.named_parameters())
        assert all(torch.equal(v,vq.state_dict()[n].detach().cpu()) for n,v in frozen.items())
        assert any(not torch.equal(v,dict(model.named_parameters())[n].detach().cpu()) for n,v in pixel_initial.items())
        torch.save(dict(state_dict=model.state_dict(),config=OmegaConf.to_container(pc,resolve=True),qualification=False,updates=c['pixel_updates']),OUT/'pixel.pt')
        assert all(sha(p)==h for p,h in manifest.items())
        assert set(reads)=={str(p.resolve()) for p in files} and len(denied)==3
        result.update(passed=True,stages=stages,parameter_inventory=counts,kmeans_seconds=kmeans_seconds,
            loading_seconds=loading,cpu_rgb_bytes=sum(d.nbytes for d in data),training_episodes=32,source_files=len(manifest),
            source_hashes_unchanged=True,lossless_tiling=True,action_arrays_read=0,simulator_queries=0,
            negative_access_checks=denied,frozen_vq_arrays=len(frozen),vq_frozen_during_pixel=True,
            checkpoint_hashes={s:sha(OUT/(s+'.pt')) for s in ('vq','pixel')},
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
            measurement='Synchronized end-to-end update, including CPU assembly, transfers and gradient checks. Steady updates21 through final update. Kmeans separately timed. Allocator peaks are not total GPU memory.',
            limitations=['Full fixed2000 VQ/20000 pixel updates per seed; upstream completion alone does not establish control performance',
                         'Adjacent frame offset1 and 128 tiling are task adapters; not the full author walker recipe',
                         'Export/history/grounding/evaluation are separate downstream stages'])
        save('status.json',dict(state='complete',**identity))
    except BaseException:
        result['traceback']=traceback.format_exc();save('status.json',dict(state='failed',**identity));raise
    finally:
        result['active_seconds']=time.perf_counter()-start;save('result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='stages'},ensure_ascii=True))

if __name__=='__main__':main()
