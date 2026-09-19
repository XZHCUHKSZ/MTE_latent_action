"""Real seven-path preflight before locking the new fine-tuning pilot."""
import concurrent.futures
import os
import subprocess
import sys
from experiments import policy_finetune_pilot as E


def main():
    out=E.PACKAGE/'outputs/policy_finetune_pilot_preflight_2026_09_20'
    out.mkdir(exist_ok=False)
    config=E.read(E.PACKAGE/'configs/policy_finetune_pilot.json')
    def run(job):
        path=out/('__'.join(map(str,job))+'.log')
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with path.open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,'-X','utf8','-m','experiments.policy_finetune_pilot','worker','--out',str(out),'--job',*map(str,job)],cwd=E.PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,(job,path.read_text(encoding='utf-8'))
        E.validate(out,job)
        return dict(job=job,passed=True)
    completed=[]
    for kind in ['ground','evaluate']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            completed+=list(pool.map(run,E.jobs(config,'smoke',kind)))
        print('Real smoke passed:',kind,flush=True)
    E.atomic_json(out/'CHECKS.json',dict(passed=True,checks=completed))
    print('PASS: seven real training paths and seven original-environment single-episode rollouts',flush=True)


if __name__=='__main__':main()
