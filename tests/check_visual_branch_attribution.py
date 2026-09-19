"""Actual branch-freeze, native reproduction and environment preflight."""
import concurrent.futures
import os
import subprocess
import sys
from experiments import visual_branch_attribution as E


def main():
    out=E.PACKAGE/'outputs/visual_branch_attribution_preflight_2026_09_20'
    out.mkdir(exist_ok=False)
    c=E.read(E.PACKAGE/'configs/visual_branch_attribution.json')
    E.snapshot(out,c)
    def run(job):
        path=out/('__'.join(map(str,job))+'.log')
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with path.open('x',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-X','utf8','-m','experiments.visual_branch_attribution','worker','--out',str(out),'--job',*map(str,job)],cwd=E.PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert proc.returncode==0,(job,path.read_text(encoding='utf-8'))
        E.validate(out,job);return dict(job=job,passed=True)
    checks=[]
    for phase in ['smoke','parity']:
        for kind in ['ground','evaluate']:
            work=[j for j in E.jobs(c,phase,kind) if j[2]==c['seeds'][0]]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks+=list(pool.map(run,work))
            print('PASS actual paths',phase,kind,len(work),flush=True)
    import torch
    for job in [j for j in E.jobs(c,'parity','ground') if j[2]==c['seeds'][0]]:
        _,_,seed,arm,_=job;f=arm.split('_')[0];mode=arm.split('_')[-1]
        old,result=E.reference(c,seed,f,mode);new=E.location(out,job)
        a=torch.load(new/'decoder.pt',map_location='cpu',weights_only=False);b=torch.load(old/'decoder.pt',map_location='cpu',weights_only=False)
        error=max(float((a['state_dict'][k]-b['state_dict'][k]).abs().max()) for k in b['state_dict']);assert error<=1e-6,(job,error)
        assert E.np.max(E.np.abs(E.np.load(new/'fit_predictions.npy')-E.np.load(old/'fit_predictions.npy')))<=1e-6
        if mode=='trainable':assert all(torch.equal(a['history_state_dict'][k],b['history_state_dict'][k]) for k in b['history_state_dict'])
        with E.np.load(E.location(out,(*job[:-1],'evaluate'))/'episode_908601.npz') as x,E.np.load(result.parent/'episode_908601.npz') as y:
            assert all(E.np.array_equal(x[k],y[k]) for k in y.files)
        checks.append(dict(job=job,decoder_max_abs=error,passed=True))
    E.verify_sources(out)
    E.atomic_json(out/'CHECKS.json',dict(passed=True,checks=checks))
    print('PASS: ten selective-gradient paths, ten frozen/joint reproductions, twenty original-environment replays',flush=True)


if __name__=='__main__':main()
