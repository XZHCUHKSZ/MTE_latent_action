"""Real nine-path preflight; no toy replacement of histories or environments."""
import concurrent.futures
import os
import subprocess
import sys
from experiments import mte_finetune_extension as E


def main():
    out=E.PACKAGE/'outputs/mte_finetune_extension_preflight_2026_09_20'
    out.mkdir(exist_ok=True)
    c=E.read(E.PACKAGE/'configs/mte_finetune_extension.json')
    E.wait_dependency(out,c)
    def run(job):
        path=out/('__'.join(map(str,job))+'.log')
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with path.open('x',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-X','utf8','-m','experiments.mte_finetune_extension','worker','--out',str(out),'--job',*map(str,job)],cwd=E.PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert proc.returncode==0,(job,path.read_text(encoding='utf-8'))
        E.validate(out,job)
        return dict(job=job,passed=True)
    checks=[]
    for kind in ['ground','evaluate']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            checks+=list(pool.map(run,E.jobs(c,'smoke',kind)))
        print('PASS nine actual paths:',kind,flush=True)
    import torch
    parity=[j for j in E.jobs(c,'parity','ground') if j[1]=='mamujoco' and j[2]==0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:checks+=list(pool.map(run,parity))
    for job in parity:
        g,r=E.reference(c,*job[1:4]);a=torch.load(E.location(out,job)/'decoder.pt',map_location='cpu',weights_only=False)['decoder']
        b=torch.load(g,map_location='cpu',weights_only=False)
        error=max(float((a[k]-b[k]).abs().max()) for k in b);assert error<=1e-6,(job,error)
        checks.append(dict(job=job,decoder_max_abs=error,passed=True))
    for job in parity:
        ej=(*job[:-1],'evaluate');checks.append(run(ej))
        _,old=E.reference(c,*job[1:4]);a=E.read(E.location(out,ej)/'result.json');b=E.read(old)['evaluation']
        assert a['episode_seeds']==b['episode_seeds'][:1]
        assert E.np.allclose(a['episode_returns'],b['episode_returns'][:1],rtol=0,atol=1e-9)
    E.atomic_json(out/'CHECKS.json',dict(passed=True,checks=checks))
    print('PASS nine actual training/rollout paths; three full MaMuJoCo frozen weight and original return parities',flush=True)


if __name__=='__main__':main()
