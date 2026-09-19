"""Real GPU/rollout qualification; run once in a new preflight directory."""
import os
import shutil
import subprocess
import sys
from experiments import visual_low_label as E


def main():
    out=E.PACKAGE/'outputs/visual_low_label_preflight_2026_09_19'
    out.mkdir(exist_ok=False)
    assets=out/'assets';(assets/'labels/dev').mkdir(parents=True)
    shutil.copy2(E.PIXEL/'pilot_protocol.json',assets/'visual_config.json')
    for i in range(8):shutil.copy2(E.PIXEL/f'runtime/pilot/validate/target_labels/dev/{i:04d}.npy',assets/f'labels/dev/{i:04d}.npy')
    p=E.read(E.PACKAGE/'configs/visual_low_label.json')
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    jobs=E.phases(p)[0][1]
    for arm in [p['main_model'],'bc_batch256','bc_idm_relabel']:
        jobs += [('parity',p['seeds'][0],8,arm,k) for k in ['ground','evaluate']]
    for job in jobs:
        print('CHECK',job,flush=True)
        log=out/('__'.join(map(str,job))+'.log')
        with log.open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,'-X','utf8','-m','experiments.visual_low_label','worker','--out',str(out),'--job',*map(str,job)],cwd=E.PACKAGE,env=env,stdout=f,stderr=subprocess.STDOUT)
        assert r.returncode==0,log.read_text(encoding='utf-8')
        E.validate_result(out,job)
    E.atomic_json(out/'qualification.json',dict(passed=True,smokes=6,b8_parity=3,replays=3,scope='three representative training paths; all45 formal gates still required'))
    print('PREFLIGHT PASS',flush=True)


if __name__=='__main__':main()
