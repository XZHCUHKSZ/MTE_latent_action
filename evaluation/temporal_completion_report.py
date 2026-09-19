"""Read-only aggregation of explicitly separated reused and new evidence."""
import json
from pathlib import Path
import numpy as np
from evaluation.temporal_scale_report import paired_stats,normalized_auc,holm
from utils.atomic import atomic_json


def summarize(out,c):
    from experiments.mpe_evidence_completion import plan
    out=Path(out);seeds=plan(c,c['full_n'])['seeds'];new=[];reused=[]
    for file in Path(c['source_run']).glob('seed*/grounding/budget_*/*.json'):
        if file.name in ('results.json','complete.json'):continue
        r=json.loads(file.read_text())
        if r['seed'] in seeds and r['budget'] in c['budgets']:reused.append(dict(r,n=c['full_n'],evidence='reused'))
    expected=0;diagnostics={}
    for n in [c['full_n'],*c['scales']]:
        p=plan(c,n);expected+=len(seeds)*len(p['budgets'])*len(p['control_configs'])
        for seed in seeds:
            for b in p['budgets']:
                for method in p['control_configs']:
                    path=out/f'n{n}'/f'seed{seed}'/'grounding'/f'budget_{b}'/f'b{b}_{method.replace("+","__")}.json'
                    if path.exists():
                        r=json.loads(path.read_text());assert (r['seed'],r['budget'],r['method'])==(seed,b,method)
                        assert r['episode_seeds']==list(range(p['evaluation_seed_start'],p['evaluation_seed_start']+p['evaluation_episodes']))
                        assert np.isclose(r['mean_return'],np.mean(r['episode_returns']))
                        new.append(dict(r,n=n,evidence='new'))
            if n==c['full_n']:
                for name in ('probes','donor_effects'):
                    path=out/f'n{n}'/f'seed{seed}'/f'{name}.json'
                    if path.exists():diagnostics.setdefault(name,[]).append(seed)
    allrows=reused+new;lookup={(r['n'],r['seed'],r['budget'],r['method']):r for r in allrows}
    assert len(lookup)==len(allrows),'Duplicate protocol identity'
    means=[]
    for n,b,m in sorted({(r['n'],r['budget'],r['method']) for r in allrows}):
        rs=[r for r in allrows if (r['n'],r['budget'],r['method'])==(n,b,m)]
        means.append(dict(n=n,budget=b,method=m,seeds=len(rs),return_mean=float(np.mean([r['mean_return'] for r in rs])),
                          return_sd=float(np.std([r['mean_return'] for r in rs],ddof=1)) if len(rs)>1 else None))
    contrasts=[]
    for a,b in c['new_primary_contrasts']:
        values=[]
        for s in seeds:
            if all((c['full_n'],s,k,m) in lookup for k in c['budgets'] for m in (a,b)):
                v=[lookup[c['full_n'],s,k,a]['mean_return']-lookup[c['full_n'],s,k,b]['mean_return'] for k in c['budgets']]
                values.append(dict(seed=s,delta=normalized_auc(v,c['budgets'])))
        contrasts.append(dict(kind='new_structure_auc',plus=a,minus=b,values=values,**paired_stats([v['delta'] for v in values])))
    for m in c['scaling_control_configs']:
        values=[]
        for s in seeds:
            keys=[(n,s,c['scaling_budget'],m) for n in (c['full_n'],min(c['scales']))] if c['scales'] else []
            if keys and all(k in lookup for k in keys):
                values.append(dict(seed=s,delta=lookup[keys[0]]['mean_return']-lookup[keys[1]]['mean_return']))
        contrasts.append(dict(kind='unlabelled_full_minus_small',method=m,values=values,**paired_stats([v['delta'] for v in values])))
    complete=len(new)==expected
    for kind in ('new_structure_auc','unlabelled_full_minus_small'):
        group=[r for r in contrasts if r['kind']==kind]
        if complete and all(len(r['values'])==len(seeds) and r['p'] is not None for r in group):
            for r,p in zip(group,holm([r['p'] for r in group])):r['holm_p']=p
        else:
            for r in group:r.pop('p',None);r.pop('ci95',None);r['inference']='pending complete matrix or non-estimable variance'
    stats=dict(new_control_rows=len(new),expected_new_control_rows=expected,reused_control_rows=len(reused),diagnostics=diagnostics)
    atomic_json(out/'summary.json',dict(**stats,means=means,contrasts=contrasts))
    lines=['# MPE补齐实验进度与结果','',f'新增 {len(new)}/{expected}；复用 {len(reused)} 条原结果。','',
        'Edge-CARA/Tree使用原始冻结训练函数；Simple/Graph/MIF沿用已声明route目标。两者不冒称等损失的结构配对。',
        '探针是冻结模型开发诊断；物理评估不回流训练。旧证据不覆盖，未完成时不作显著性结论。','',
        '|N|B|模型|种子|回报 ↑|','|---:|---:|---|---:|---:|']
    for r in means:lines.append(f"|{r['n']}|{r['budget']}|{r['method']}|{r['seeds']}|{r['return_mean']:.5f}|")
    lines+=['','## 固定配对','']+[json.dumps(r,ensure_ascii=False) for r in contrasts]
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return stats
