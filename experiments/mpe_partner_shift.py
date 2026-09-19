"""Evaluate a locked model matrix under partner-policy shifts; never retrain."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import t
from evaluation.mpe_partner_shift import restore_grounded,evaluate,digest
from utils.atomic import atomic_json

PACKAGE=Path(__file__).resolve().parents[1]


def worker(task):
    row,config,out=task;torch.set_num_threads(1)
    path=Path(row['source']).with_suffix('.pt');before=digest(path)
    policy=restore_grounded(path,row['method'],config['budget'])
    old=json.loads(Path(row['source']).read_text())
    replay=evaluate(policy,old['episode_seeds'][:2],'identity')
    assert np.allclose([r['return_'] for r in replay['episodes']],old['episode_returns'][:2],rtol=0,atol=1e-7), 'Identity replay does not match archived policy'
    seeds=range(config['episode_seed_start'],config['episode_seed_start']+config['episodes'])
    rows=[dict(seed=row['seed'],method=row['method'],**evaluate(policy,seeds,condition))
          for condition in config['conditions']]
    assert before==digest(path)
    result=dict(rows=rows,weight_sha256=before,source=row['source'],identity_replay_pass=True,
                training_updates=0,weights_unchanged=True)
    atomic_json(Path(out)/f"seed{row['seed']}_{row['method'].replace('+','__')}.json",result)
    return result


def main(smoke=False):
    config=json.loads((PACKAGE/'configs/mpe_partner_shift.json').read_text())
    if smoke:
        config.update(training_seeds=[45],methods=['base+graph','bc'],episodes=2,
                      episode_seed_start=97919000,output=config['output']+'_smoke')
    out=PACKAGE/config['output'];out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',config)
    inventory=json.loads((PACKAGE/config['inventory']).read_text())
    tasks=[r for r in inventory['records'] if r['n']==config['n'] and r['budget']==config['budget']
           and r['seed'] in config['training_seeds'] and r['method'] in config['methods']]
    for method,template in config.get('structural_sources',{}).items():
        if method not in config['methods']:continue
        for seed in config['training_seeds']:
            assert not any(r['method']==method and r['seed']==seed for r in tasks)
            path=PACKAGE/template.format(seed=seed)
            r=json.loads(path.read_text())
            assert r['method']==method and r['seed']==seed and r['budget']==config['budget']
            tasks.append(dict(r,n=config['n'],source=str(path)))
    assert len(tasks)==len(config['training_seeds'])*len(config['methods'])
    # Lock weight identities and implementation hashes before any shifted rollout.
    files=[Path(__file__),PACKAGE/'evaluation/mpe_partner_shift.py',PACKAGE/'training/composition.py',
           PACKAGE/'training/grounding_mpe.py',PACKAGE/'closed_loop_lam_v1/common.py',
           PACKAGE/'environments/mpe.py',PACKAGE/'environments/native_particle.py']
    atomic_json(out/'source_manifest.json',dict(code={str(f.relative_to(PACKAGE)):digest(f) for f in files},
        weights={r['source']:digest(Path(r['source']).with_suffix('.pt')) for r in tasks}))
    start=time.time();completed=[]
    atomic_json(out/'status.json',dict(status='running',completed=0,total=len(tasks)))
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(worker,(r,config,str(out))) for r in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
            atomic_json(out/'status.json',dict(status='running',completed=len(completed),total=len(tasks),elapsed_seconds=time.time()-start))
    rows=[r for c in completed for r in c['rows']]
    means=[dict(method=m,condition=c,mean=float(np.mean([r['mean_return'] for r in rows if r['method']==m and r['condition']==c])))
           for m in config['methods'] for c in config['conditions']]
    contrasts=[]
    if not smoke:
        for comparator in config['primary_comparators']:
            differences=[]
            for seed in config['training_seeds']:
                score=lambda method:np.mean([r['mean_return'] for r in rows if r['seed']==seed and r['method']==method and r['condition']!='identity'])
                differences.append(float(score(config['primary_method'])-score(comparator)))
            values=np.array(differences);mean=values.mean();se=values.std(ddof=1)/np.sqrt(len(values))
            flips=np.array(list(product([-1,1],repeat=len(values))))
            pv=float(np.mean(np.abs((flips*values).mean(1))>=abs(mean)-1e-12))
            contrasts.append(dict(comparator=comparator,mean=float(mean),by_seed=differences,
                ci95=[float(mean-t.ppf(.975,len(values)-1)*se),float(mean+t.ppf(.975,len(values)-1)*se)],p_exact=pv))
        running=0.
        for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['p_exact'])):
            running=max(running,min(1.,(len(contrasts)-rank)*contrasts[i]['p_exact']));contrasts[i]['p_holm']=running
    seeds=range(config['episode_seed_start'],config['episode_seed_start']+config['episodes'])
    anchors=[dict(anchor=a,**evaluate(None,seeds,c,a)) for a in ('teacher','random') for c in config['conditions']]
    manifest=json.loads((out/'source_manifest.json').read_text())
    assert all(digest(PACKAGE/f)==v for f,v in manifest['code'].items())
    assert all(digest(Path(f).with_suffix('.pt'))==v for f,v in manifest['weights'].items())
    atomic_json(out/'summary.json',dict(protocol=config,means=means,rows=rows,primary_contrasts=contrasts,
        anchors=anchors,weights_unchanged=True,code_unchanged=True,training_updates=0))
    report=['# MPE冻结模型伙伴策略变化评估','',
       'N700、B32；同一冻结模型，不训练、不调参。伙伴动作保持原样、减半、旋转±30度。',
       '这检验受控策略变化，不等同于新环境、新伙伴身份或新增训练种子。','',
       '|方法|原策略|减速|旋转+30°|旋转−30°|','|---|---:|---:|---:|---:|']
    for method in config['methods']:
        v=[next(r['mean'] for r in means if r['method']==method and r['condition']==c) for c in config['conditions']]
        report.append('|'+method+'|'+'|'.join(f'{x:.4f}' for x in v)+'|')
    report+=['','## 预定Graph-Aux主对比：三种变化等权、种子内先汇总','',
             '|对照|均值差|95%配对区间|精确p经Holm|','|---|---:|---|---:|']
    for c in contrasts:report.append(f"|{c['comparator']}|{c['mean']:+.4f}|[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]|{c['p_holm']:.4f}|")
    report+=['','五个种子为复制单位；episode/三种变化不是额外独立训练种子。',
       '区间为t区间，p值为精确符号置换；五个单位的双侧精确p不可能小于0.0625。',
       'Teacher/random难度参照与所有逐episode数据在summary.json；不筛选条件或种子。']
    (out/'RESULTS_CN.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    atomic_json(out/'status.json',dict(status='complete',completed=len(tasks),total=len(tasks),
           all_identity_replays_pass=True,weights_unchanged=True,elapsed_seconds=time.time()-start))
    print('Complete',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();main(args.smoke)
