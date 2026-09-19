"""Read-only checks against an explicitly supplied accepted manuscript.

Checks main return tables at their printed precision, figure input seed values,
and all table/source bindings. Does not rewrite the paper or claim to verify proofs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics as st


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


MPE = {'Base':'base','Base-duplicate':'base_duplicate','Entity-joint':'entity_joint',
       'LAPO-joint':'lapo','LAOM-joint':'laom','LAPO-state':'lapo_state','LAOM-state':'laom_state',
       'BC':'bc','IDM-relabel':'idm','LAOM-state-Aux':'base+laom_state','Global-Joint16-Aux':'base+global16'}
MA = {'Continuous':'continuous_lam','LAPO-state':'lapo_state_adapter','LAOM-state':'laom_state_adapter',
      'Edge-CARA':'edge_cara','Edge-h1':'edge_cara_h1','Simple':'edge_cara_mobius_simple',
      'Graph':'edge_cara_mobius_graph','Tree':'edge_cara_mobius_tree','MIF':'edge_cara_mif',
      'DeepSets':'matched_cf_deepsets','Random-edge':'random_edge_encoder','LAPO-joint':'lapo_joint',
      'LAOM-joint':'laom_joint_k3','All-slots':'entity_joint','Target-slot':'entity_target',
      'Base':'anchor_only','Base-duplicate':'anchor_duplicate','BC':'bc_scarce','IDM-relabel':'idm_relabel',
      'LAOM-state-Aux':'anchor_laom'}
VIS = {'Base':'anchor_only','Base-duplicate':'anchor_duplicate','Feature-BC-256':'bc_batch256',
       'Feature-BC-1600':'bc_full1600','IDM-relabel':'bc_idm_relabel'}
for label, key in [('Edge-h2','edge_h2'),('Edge-h3','edge_h3'),('Simple','simple'),('Graph','graph'),
                   ('Tree','tree'),('MIF','mif'),('DeepSets','deepsets'),('Random-edge','random'),('Continuous','continuous')]:
    MPE[label+'-Solo']=key
    MPE[label+'-Aux']='base+'+key
for label, key in [('Edge-h1','edge_h1'),('Edge-h3','edge_h3'),('Simple','simple'),('Graph','graph'),
                   ('Tree','tree'),('MIF','mif'),('DeepSets','matched'),('Random-edge','random_edge'),('Continuous','continuous')]:
    MA[label+'-Aux']='anchor_'+key
for label, key in [('Edge-CARA','edge_cara'),('Simple','edge_cara_mobius_simple'),('Graph','edge_cara_mobius_graph'),
                   ('Tree','edge_cara_mobius_tree'),('MIF','edge_cara_mif'),('DeepSets','matched_cf_deepsets'),
                   ('Random-edge','random_edge_encoder'),('Continuous','continuous_lam'),
                   ('LAPO-state','lapo_state_adapter'),('LAOM-state','laom_state_adapter')]:
    VIS[label+'-Solo']='anchor_solo_'+key
    VIS[label+'-Aux']='anchor_plus_'+key

# Figure aliases are explicit: old display ID edge_h1 denotes repaired MPE h2.
FIG_MPE = {'entity_target':'base','entity_joint':'entity_joint','anchor_only':'base','anchor_duplicate':'base_duplicate',
           'lapo_joint':'lapo','laom_joint_k3':'laom','bc_scarce':'bc','idm_relabel':'idm',
           'continuous_lam':'continuous','lapo_state_adapter':'lapo_state','laom_state_adapter':'laom_state',
           'anchor_laom':'base+laom_state','edge_cara_h1':'edge_h2','edge_cara':'edge_h3',
           'edge_cara_mobius_simple':'simple','edge_cara_mobius_graph':'graph','edge_cara_mobius_tree':'tree',
           'edge_cara_mif':'mif','matched_cf_deepsets':'deepsets','random_edge_encoder':'random'}
for old, new in [('edge_h1','edge_h2'),('edge_h3','edge_h3'),('simple','simple'),('graph','graph'),
                 ('tree','tree'),('mif','mif'),('matched','deepsets'),('random_edge','random'),('continuous','continuous')]:
    FIG_MPE['anchor_'+old]='base+'+new


def check(manuscript, release):
    e=manuscript/'evidence'; text=(manuscript/'main.tex').read_text(encoding='utf-8')
    lock=read(release/'provenance/manuscript_snapshot.json')
    for r in lock['files']:
        assert sha(manuscript/r['path'])==r['sha256'], ('different manuscript',r['path'])
    assert lock['adaptation_integrated'] is False
    mpe=read(release/'results/completed_runs/mpe_inventory_completion_2026_09_18/summary.json')['records']
    ma=read(release/'results/current/mamujoco_controls.json')
    vis=read(release/'results/completed_runs/visual_low_label_2026_09_19_r1/summary.json')['records']
    vis8=read(e/'visual_summary.json')['rows']
    numeric_checks=[]
    tables=list(re.finditer(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',text,re.S))
    assert len(tables)==21, 'Table inventory changed; review source bindings'
    for index,m in enumerate(tables[:5]):
        env='mpe' if index<2 else 'mamujoco' if index<4 else 'visual'
        mapping=MPE if env=='mpe' else MA if env=='mamujoco' else VIS
        budgets=[8,16,32,64,128,256] if env=='mpe' else [4,8,16,32,64] if env=='mamujoco' else [1,2,4,8]
        digits=3 if env=='mpe' else 2
        for line in m.group(1).splitlines():
            cells=line.split('&'); label=cells[0].strip()
            if label not in mapping:continue
            assert len(cells)==1+len(budgets),label
            for b,cell in zip(budgets,cells[1:]):
                numbers=re.findall(r'-?\d+\.\d+',cell)
                if not numbers:
                    assert env=='visual' and '--' in cell
                    continue
                assert len(numbers)==2,(label,cell)
                key=mapping[label]
                if env=='mpe':
                    y=[r['mean_return'] for r in sorted(mpe,key=lambda r:r['seed']) if (r['n'],r['budget'],r['method'])==(700,b,key)]
                elif env=='mamujoco':
                    y=next(r['seed_returns'] for r in ma if (r['n'],r['budget'],r['method'])==(90,b,key))
                elif b==8:
                    y=next(r['training_seed_returns'] for r in vis8 if r['arm']==key)
                else:
                    y=[r['mean_return'] for r in sorted(vis,key=lambda r:r['seed']) if (r['budget'],r['arm'])==(b,key)]
                assert len(y)==5,(env,label,b)
                expected=[f'{st.mean(y):.{digits}f}',f'{st.stdev(y):.{digits}f}']
                assert expected==numbers,(env,label,b,expected,numbers)
                numeric_checks.append(dict(table=index+1,method=label,budget=b,mean=numbers[0],sd=numbers[1]))
    assert len(numeric_checks)==371,len(numeric_checks)
    figure_points=0
    for r in read(e/'figure_numeric_current.json'):
        if r['environment']=='mpe':
            method=FIG_MPE[r['method']]
            y=[v['mean_return'] for v in sorted(mpe,key=lambda v:v['seed']) if (v['n'],v['budget'],v['method'])==(r['n'],r['budget'],method)]
        else:
            y=next(v['seed_returns'] for v in ma if (v['n'],v['budget'],v['method'])==(r['n'],r['budget'],r['method']))
        assert len(y)==5 and max(abs(x-z) for x,z in zip(y,r['seed_returns']))<1e-10,r
        figure_points+=len(y)
    for r in read(e/'figure_scaling_current.json'):
        if r['environment']=='mpe':
            y=[[next(v['mean_return'] for v in mpe if (v['n'],v['budget'],v['method'],v['seed'])==(n,r['budget'],FIG_MPE[r['method']],s)) for n in r['n_values']] for s in range(45,50)]
        else:
            y=next(v['seed_returns_by_n'] for v in read(e/'unchanged_numeric_scaling.json') if (v['environment'],v['method'])==('mamujoco',r['method']))
        assert y==r['seed_returns_by_n'],r['method']
        figure_points+=sum(len(v) for v in y)
    routes=read(e/'control_ablation_current.json')
    for r in routes['mpe']:
        y=[st.mean(v['mean_return'] for v in mpe if v['n']==700 and v['seed']==s and v['method']=='base+'+r['method'])-st.mean(v['mean_return'] for v in mpe if v['n']==700 and v['seed']==s and v['method']=='base+global16') for s in range(45,50)]
        assert max(abs(x-z) for x,z in zip(y,r['upstream_seed_mean_gains']))<1e-10
        figure_points+=5
    r=routes['mamujoco']; raw=read(release/'results/completed_runs/mamujoco_route_completion_2026_09_19/summary.json')['rows']
    for i,s in enumerate(r['upstream_seed_ids']):
        for j,b in enumerate(r['budgets']):
            delta=st.mean(v['mean_return'] for v in raw if (v['seed'],v['budget'],v['arm'])==(s,b,'mif_matched'))-st.mean(v['mean_return'] for v in raw if (v['seed'],v['budget'],v['arm'])==(s,b,'global16'))
            assert abs(delta-r['upstream_seed_mean_gains_by_budget'][i][j])<1e-10
            figure_points+=1
    figure_artifacts=[]
    for relative in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
        figure_artifacts.append(dict(path=relative,sha256=sha(manuscript/relative)))
    assert len(figure_artifacts)==3
    evidence=[dict(path=f.name,sha256=sha(f)) for f in sorted(e.iterdir()) if f.is_file() and f.suffix in ('.json','.tex','.md')]
    return dict(status='pass',manuscript=lock,table_count=len(tables),main_return_table_count=5,
                checked_mean_sd_cells=len(numeric_checks),checked_figure_seed_values=figure_points,
                table_inventory=[dict(index=i+1,line=text[:m.start()].count('\n')+1,
                    labels=re.findall(r'\\label\{([^}]+)',m.group(1)),
                    block_sha256=hashlib.sha256(m.group(0).encode()).hexdigest()) for i,m in enumerate(tables)],
                main_table_cells=numeric_checks,figure_artifacts=figure_artifacts,evidence=evidence,
                figure_script=dict(path='scripts/draw_reference_composites.py',sha256=sha(manuscript/'scripts/draw_reference_composites.py')),
                recorded_rgb=dict(path='evidence/illustration_rgb.npy',sha256=sha(e/'illustration_rgb.npy'),distributed=False),
                scope='Five main return tables recomputed at printed precision; figure numeric/scaling/route input values checked; all 21 table blocks indexed. Secondary table values, mathematical proofs and pixel-level layout are not independently rederived by this checker.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manuscript',required=True,type=Path)
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    a=p.parse_args();r=check(a.manuscript,a.release)
    with a.report.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:r[k] for k in ('status','table_count','checked_mean_sd_cells','checked_figure_seed_values')}))
