"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import argparse,json

from pathlib import Path

from visual import train as V

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,required=True,choices=range(908711,908716))
    ap.add_argument('--run-dir',required=True);ap.add_argument('--data-dir',required=True)
    ap.add_argument('--asset-dir',required=True,help='visual_config.json and labels/{train,dev}; labels opened only after freeze')
    ap.add_argument('--stage',required=True,choices=['frontend','features','bridge','rich','pretrain','freeze','ground','evaluate'])
    ap.add_argument('--method');ap.add_argument('--global-freeze')
    a=ap.parse_args();V.configure(a.seed,a.run_dir,a.data_dir,a.asset_dir)
    # Preserve the final five-seed worker's process-level determinism settings.
    V.torch.set_num_threads(1);V.torch.set_num_interop_threads(1);V.C.seed_all(a.seed)
    V.torch.use_deterministic_algorithms(True)
    V.torch.backends.cudnn.benchmark=False;V.torch.backends.cudnn.deterministic=True
    V.torch.backends.cudnn.allow_tf32=False;V.torch.backends.cuda.matmul.allow_tf32=False
    if a.stage in {'ground','evaluate'}:
        if not a.global_freeze:ap.error('Pass an all-five-seed freeze record')
        from training.freeze import verify
        verify(a.global_freeze)
    name={'pretrain':'pre_'+str(a.method),'ground':'ground_'+str(a.method),'evaluate':'evaluate_'+str(a.method)}.get(a.stage,a.stage)
    out=Path(a.run_dir)/name;out.mkdir(exist_ok=False)
    if a.stage=='pretrain':
        if a.method not in V.METHODS:ap.error('Unknown paper method')
        r=V.pretrain_base(out,a.method) if a.method=='visual_laom_target' else V.pretrain(out,a.method)
    elif a.stage=='ground':
        if a.method not in V.ARMS:ap.error('Unknown paper control')
        r=V.idm(out) if a.method=='bc_idm_relabel' else V.ground(out,a.method)
    elif a.stage=='evaluate':
        from visual import rollout
        rollout.RT=V.RT;r=rollout.evaluate(out,a.method)
    else:r=getattr(V,a.stage)(out)
    V.put(out/'result.json',r)

if __name__=='__main__':main()
