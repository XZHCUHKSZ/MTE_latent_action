"""Real CUDA smoke and frozen-route numerical reproduction, isolated workers."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments import visual_supervision_inventory as E

def main():
    out=E.PACKAGE/'outputs/visual_supervision_inventory_preflight_2026_09_19_r1'
    out.mkdir(exist_ok=False)
    p=E.read(E.PACKAGE/'configs/visual_supervision_inventory.json')
    phases=E.phases(p)
    assert len(phases[0][1])==8 and len(phases[1][1])==40
    assert len(phases[2][1])==len(phases[3][1])==80
    assert len(phases[4][1])==len(phases[5][1])==360
    jobs=phases[0][1]+[('fair_parity',p['seeds'][0],1,f'mif_{c}_pretrained_frozen','ground') for c in ['solo','aux']]
    def run(j):
        log=out/('__'.join(map(str,j))+'.log')
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with log.open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,'-X','utf8','-m','experiments.visual_supervision_inventory','worker','--out',str(out),'--job',*map(str,j)],cwd=E.PACKAGE,env=env,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,log.read_text(encoding='utf-8')
        E.validate_result(out,j)
        return dict(job=j,passed=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(run,jobs))
    (out/'CHECKS.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,real_workers=len(results),out=str(out))))
if __name__=='__main__':main()
