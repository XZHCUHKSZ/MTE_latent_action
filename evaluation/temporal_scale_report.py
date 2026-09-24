"""Complete-matrix reporting; paired training seeds are the statistical units."""
from mte.method_names import report_text
from mte.method_names import normalize_config
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from utils.atomic import atomic_json


def paired_stats(values):
    values = np.asarray(values, float); n = len(values)
    if n < 2: return dict(n=n, mean=float(values.mean()) if n else None, ci95=None, p=None)
    mean = float(values.mean()); se = float(values.std(ddof=1)/np.sqrt(n))
    if se == 0:
        return dict(n=n, mean=mean, ci95=[mean, mean], p=1. if mean == 0 else None,
                    warning='Degenerate variance; no finite t-test for nonzero constant differences')
    radius = float(student_t.ppf(.975, n-1)*se)
    return dict(n=n, mean=mean, ci95=[mean-radius, mean+radius],
                p=float(2*student_t.sf(abs(mean/se), n-1)),
                assumption='Approximate normality of independent training-seed differences; n=5 is small')


def normalized_auc(values, budgets):
    log = np.log2(np.asarray(budgets, float))
    return float(np.trapezoid(np.asarray(values, float), log)/(log[-1]-log[0]))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    result = [None]*len(ps); last = 0.
    for rank, i in enumerate(order):
        last = max(last, min(1., (len(ps)-rank)*ps[i])); result[i] = last
    return result


def summarize(out, p):
    p = normalize_config(p, "mpe")
    out = Path(out); rows = []
    for seed in p['seeds']:
        for budget in p['budgets']:
            folder = out/f'seed{seed}/grounding/budget_{budget}'
            for method in p['control_configs']:
                path = folder/f'b{budget}_{method.replace("+", "__")}.json'
                if path.exists():
                    row = json.loads(path.read_text()); assert (row['seed'], row['budget'], row['method']) == (seed, budget, method)
                    assert row['episode_seeds'] == list(range(p['evaluation_seed_start'], p['evaluation_seed_start']+p['evaluation_episodes']))
                    assert len(row['episode_returns']) == p['evaluation_episodes']
                    rows.append(row)
    bykey = {(r['seed'], r['budget'], r['method']): r for r in rows}
    assert len(bykey) == len(rows)
    expected = len(p['seeds'])*len(p['budgets'])*len(p['control_configs'])
    means = []
    for b in p['budgets']:
        for method in p['control_configs']:
            found = [bykey[(s,b,method)] for s in p['seeds'] if (s,b,method) in bykey]
            if found:
                means.append(dict(budget=b, method=method, seeds=[r['seed'] for r in found],
                    mean_return=float(np.mean([r['mean_return'] for r in found])),
                    mean_collisions=float(np.mean([r['mean_collisions'] for r in found])),
                    mean_coverage=float(np.mean([r['mean_final_coverage_distance'] for r in found]))))
    primary = []
    for a,b in p['primary_contrasts']:
        differences = []
        for seed in p['seeds']:
            if all((seed,k,m) in bykey for k in p['budgets'] for m in (a,b)):
                dif = [bykey[(seed,k,a)]['mean_return']-bykey[(seed,k,b)]['mean_return'] for k in p['budgets']]
                differences.append(dict(seed=seed, auc_delta=normalized_auc(dif,p['budgets'])))
        primary.append(dict(plus=a, minus=b, seed_differences=differences,
                            **paired_stats([d['auc_delta'] for d in differences])))
    # No significance updates from an incomplete matrix; intermediate rows descriptive only.
    if len(rows) == expected and all(r['p'] is not None for r in primary):
        for r,adj in zip(primary,holm([r['p'] for r in primary])): r['holm_p'] = adj
    else:
        for r in primary: r.pop('p',None); r.pop('ci95',None); r['inference']='pending complete matrix'
    atomic_json(out/'summary.json',dict(completed_rows=len(rows),expected_rows=expected,means=means,
        primary_auc_contrasts=primary,scope=p['scope']))
    lines=['# MPE 时间修正版：五种子、六档标签预算', '', f'控制结果：{len(rows)}/{expected}。未完成时仅报告描述性结果。',
           '',p['scope'],'','|预算|方法|种子数|回报 ↑|碰撞 ↓|覆盖距离 ↓|','|---:|---|---:|---:|---:|---:|']
    for r in means:lines.append(f"|{r['budget']}|{r['method']}|{len(r['seeds'])}|{r['mean_return']:.5f}|{r['mean_collisions']:.3f}|{r['mean_coverage']:.5f}|")
    lines += ['','## 预定主比较：每种子标签曲线面积差','',
              'AUC 在 log2 标签预算上积分并归一化；重复单位为训练种子，不是预算或评估episode。五种子t区间依赖分布假设，不能保证普遍性。','']
    for r in primary:lines.append(f"- {r['plus']} − {r['minus']}: {json.dumps(r,ensure_ascii=False)}")
    (out/'RESULTS_CN.md').write_text(report_text(lines, "mpe")+'\n',encoding='utf-8')
    return len(rows), expected
