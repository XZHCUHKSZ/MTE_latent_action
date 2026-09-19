"""Read-only source validation and human-readable synthesis; no experiment edits."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
from scipy.stats import t

P=Path(__file__).resolve().parents[1]
O=P/'outputs/mpe_partner_shift_2026_09_18_r1'
R=P/'results/mpe_review_followup_2026_09_18'


def main():
    state=json.loads((O/'status.json').read_text());assert state['status']=='complete'
    d=json.loads((O/'summary.json').read_text());c=d['protocol']
    assert d['weights_unchanged'] and d['code_unchanged'] and d['training_updates']==0
    manifest=json.loads((O/'source_manifest.json').read_text())
    for f,h in manifest['code'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h
    for f,h in manifest['weights'].items():assert hashlib.sha256(Path(f).with_suffix('.pt').read_bytes()).hexdigest()==h
    files=list(O.glob('seed*.json'));assert len(files)==55
    saved=[]
    for f in files:
        x=json.loads(f.read_text());assert x['identity_replay_pass'] and x['weights_unchanged'] and x['training_updates']==0
        assert x['weight_sha256']==manifest['weights'][x['source']]
        saved+=x['rows']
    key=lambda r:(r['seed'],r['method'],r['condition'])
    rows={key(r):r for r in d['rows']}
    expected=set(itertools.product(c['training_seeds'],c['methods'],c['conditions']))
    assert len(rows)==len(d['rows'])==220 and set(rows)==expected
    assert {key(r):r for r in saved}==rows
    episode_seeds=list(range(c['episode_seed_start'],c['episode_seed_start']+c['episodes']))
    for r in list(rows.values())+d['anchors']:
        assert [e['episode_seed'] for e in r['episodes']]==episode_seeds
        for field,metric in [('mean_return','return_'),('mean_collisions','collisions'),('mean_coverage','coverage')]:
            assert np.isclose(r[field],np.mean([e[metric] for e in r['episodes']]),rtol=0,atol=1e-12)
    assert len(d['anchors'])==8
    assert {(r['anchor'],r['condition']) for r in d['anchors']}==set(itertools.product(['teacher','random'],c['conditions']))
    score=lambda seed,m:np.mean([rows[seed,m,k]['mean_return'] for k in c['conditions'] if k!='identity'])
    contrasts=d['primary_contrasts'];assert [r['comparator'] for r in contrasts]==c['primary_comparators']
    for r in contrasts:
        v=np.array([score(s,c['primary_method'])-score(s,r['comparator']) for s in c['training_seeds']])
        assert np.allclose(v,r['by_seed'],rtol=0,atol=1e-12)
        assert np.isclose(v.mean(),r['mean'],rtol=0,atol=1e-12)
        half=t.ppf(.975,4)*v.std(ddof=1)/np.sqrt(5)
        assert np.allclose([v.mean()-half,v.mean()+half],r['ci95'])
        flips=np.array(list(itertools.product([-1,1],repeat=5)))
        p=float(np.mean(abs((flips*v).mean(1))>=abs(v.mean())-1e-12));assert p==r['p_exact']
    maximum=0
    for k,r in enumerate(sorted(contrasts,key=lambda r:r['p_exact'])):
        maximum=max(maximum,min(1,(5-k)*r['p_exact']));assert r['p_holm']==maximum
    for row in d['means']:
        value=np.mean([rows[s,row['method'],row['condition']]['mean_return'] for s in c['training_seeds']])
        assert np.isclose(value,row['mean'],rtol=0,atol=1e-12)
    report=['# MPE补证完整报告：伙伴变化、物理交互时间与donor误差','',
      '本轮已完成并独立核验。55个模型—种子任务、220条件单元、14,080条策略评估episode；另有8组teacher/random参照共512条episode。',
      'N700、B32，使用原训练seed45–49的冻结checkpoint；四种伙伴条件均在相同64个新episode初始化上评估，每次23步。没有新增训练、调参或按结果改模型。','',
      '## 1. 控制：多种家族成员有竞争力，Graph的单操作优势未得到确认','',
      '|模型|原伙伴|减速0.5|旋转+30°|旋转−30°|三种变化平均|','|---|---:|---:|---:|---:|---:|']
    for method in c['methods']:
        vals=[float(np.mean([rows[s,method,k]['mean_return'] for s in c['training_seeds']])) for k in c['conditions']]
        report.append('|'+method+'|'+'|'.join(f'{v:.4f}' for v in vals+[float(np.mean(vals[1:]))])+'|')
    report+=['','回报越高越好；这里是固定B32的新episode集合，不是原六档预算曲线的重估。',
      'Graph-Aux在三种变化平均上超过Base、Global-Joint16和LAOM，但没有超过Graph-raw或LAPO。',
      'Tree、Simple与Edge-h3在若干条件下较强；这些为预列次要模型，不能事后替换Graph成为本轮预指定主模型。','',
      '### Graph-Aux的五个预定主比较','',
      '|对照|平均差|95%配对t区间|正向种子|精确p|Holm p|','|---|---:|---|---:|---:|---:|']
    for r in contrasts:
        report.append(f"|{r['comparator']}|{r['mean']:+.5f}|[{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]|{sum(x>0 for x in r['by_seed'])}/5|{r['p_exact']:.4f}|{r['p_holm']:.4f}|")
    report+=['','主比较先在每个训练seed内平均三种变化，再做五seed配对。',
       't区间与精确符号置换使用不同假设，需同时阅读；五单位的双侧精确p最低0.0625。所有预定比较和全部方向均保留。',
       '该结果增加受控伙伴策略变化的证据；不等同于新任务、新agent身份或新训练数据泛化。','',
       '### 难度参照','', '|条件|Teacher目标|Random目标|','|---|---:|---:|']
    for condition in c['conditions']:
        vals=[next(r['mean_return'] for r in d['anchors'] if r['anchor']==a and r['condition']==condition) for a in ['teacher','random']]
        report.append('|'+condition+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
    report+=['','参照不是额外训练种子；teacher并非有保证的性能上限。不同条件的绝对回报也反映伙伴行为改变后的任务难度。','',
      '## 2. 真实物理二阶交互：短窗口存在测量边界','',
      '64个独立episode、每个4个状态和3个伙伴；同状态第一步动作保留/置零，所有分支随后动作完全相同。',
      'h1目标一阶效应为零；h2/h3的一阶位置效应非零，但四分支二阶位置效应为零。h4起出现二阶效应：RMS 0.00445；h8为0.06307，超过1e-7的配对比例分别4.2%和12.6%。',
      '这个时间边界适用于本实验的第一步动作干预、固定continuation和位置outcome，不意味着环境没有交互，也不涉及奖励的二阶效应。',
      '它提示更长时域的独立物理机制验证，但没有证明任何现有MTE模型能读出h4/h8交互；旧控制结果不因此作废。','',
      '## 3. Donor输运：问题不是只有预测幅度','',
      '只读原五种子的实际training-donor记录：h2预测/真实效应RMS比7.31–7.95，整体方向对齐约0.014–0.038；约98%预测能量在目标坐标。',
      '真实伙伴位置效应和伙伴编码变化均为零。目标donor code与query状态下相同动作重新编码的code存在差距，提示跨状态含义变化；预测器重组误差仍是另一可能来源。',
      '不能靠简单缩放或同状态结果替代该缺口。下一步应以evaluation-only重编码对照分开检查输运与读出，不把模拟器生成数据回流observation-only训练。','',
      '## 4. 尚未解决与论文同步','',
      '- History层的交互读出衰减仍未解决；需要独立训练改进与确认。',
      '- 真正的新环境/新agent身份、官方近期结构化基线、视觉低标签与camera身份、更多agent尚未补齐。',
      '- MaMuJoCo不属于本轮新增评估；其动力学时域须单独检查，不能照搬MPE的h4结论。',
      '- 论文主线、三张图、9页正文/30页PDF本轮均未修改。新伙伴变化曲线及物理时间诊断尚未写入论文；应以替换重复诊断的方式合入，而不是再堆附录。','',
      '## 5. 来源与验收','',
      '- `outputs/mpe_partner_shift_2026_09_18_r1/summary.json`、`source_manifest.json`及55份seed结果。',
      '- `outputs/mpe_physical_interaction_onset_2026_09_18/summary.json`与evaluation-only四分支数组。',
      '- `outputs/mpe_donor_transport_audit_2026_09_18/summary.json`及其原始五种子引用。',
      '- 代码7文件、权重55文件哈希一致；220条件身份唯一；每条件64episode一致；回报/coverage/collision均重新汇总；五个主比较、t区间、精确p与Holm重新计算。',
      '- 所有checkpoint已通过旧轨迹identity重放；冻结基础verifier另行通过。']
    R.mkdir(parents=True,exist_ok=True)
    (R/'RESULTS_CN.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    audit=dict(status='verified',tasks=55,condition_cells=220,policy_rollouts=14080,anchor_rollouts=512,
        code_hashes=len(manifest['code']),weight_hashes=len(manifest['weights']),identity_replays=True,
        all_metrics_recomputed=True,primary_statistics_recomputed=True,training_updates=0,
        sources={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in [O/'summary.json',O/'source_manifest.json',O/'protocol.json',O/'status.json']})
    (R/'VERIFICATION.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    doc=P/'docs/review_remaining_and_next_2026_09_18.md'
    old=doc.read_text(encoding='utf8');marker='## 本轮完成后的结论'
    if marker not in old:
        old+='\n'+marker+'\n\n55/55任务、220/220条件完成；代码/权重哈希和旧轨迹重放通过。\n'
        old+='Graph在三种变化平均上较Base、Global16、LAOM为正，较Graph-raw、LAPO为负；全部五项精确Holm比较不显著。\n'
        old+='补齐了冻结策略的受控伙伴变化检查，没有确认Möbius坐标的鲁棒性增益。\n'
        old+='完整矩阵、区间、难度参照与机制诊断见 `results/mpe_review_followup_2026_09_18/RESULTS_CN.md`。\n'
        old+='论文图表尚未同步本轮新增内容；不重写主线，不合并不同协议或偷换复制单位。\n'
        doc.write_text(old,encoding='utf8')
    print(json.dumps(audit,indent=2));print(json.dumps(contrasts,indent=2))


if __name__=='__main__':main()
