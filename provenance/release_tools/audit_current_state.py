"""Read-only audit of declared sources, weights, identities and exact statistics.

Requires the original archive. Writes only a new audit report, never repairs a
manifest or changes results. Hashes are cached across studies within this audit.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import time

HEX = re.compile(r'^[0-9a-f]{64}$')
COUNTS = {
    'mpe_inventory_completion_2026_09_18': (1305, ('n','seed','budget','method')),
    'mpe_matching_input_control_2026_09_19': (15, ('seed','method')),
    'mamujoco_route_completion_2026_09_19': (875, ('seed','arm','budget','decoder_seed')),
    'mamujoco_mif_visibility_2026_09_19': (500, ('seed','arm','budget','decoder_seed')),
    'visual_low_label_2026_09_19_r1': (180, ('seed','budget','arm')),
    'visual_supervision_inventory_2026_09_19': (620, ('seed','budget','arm')),
    'policy_finetune_pilot_2026_09_20_r1': (35, ('suite','seed','arm')),
    'mte_finetune_extension_2026_09_20': (45, ('suite','seed','arm')),
    'visual_branch_attribution_2026_09_20': (100, ('seed','family','mode')),
}


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def audit(package, catalogue):
    package=package.resolve(); workspace=package.parents[1]
    hash_cache={}; json_cache={}; failures=[]; reports=[]

    def digest(path):
        path=path.resolve()
        if path not in hash_cache:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            hash_cache[path]=h.hexdigest()
        return hash_cache[path]

    def load(path):
        path=path.resolve()
        if path not in json_cache:json_cache[path]=read(path)
        return json_cache[path]

    def checked(condition, run, kind, detail):
        if not condition:failures.append(dict(run=run,kind=kind,detail=detail))

    def references(obj, context):
        if isinstance(obj,dict):
            if isinstance(obj.get('source'),str) and Path(obj['source']).is_dir():
                context=Path(obj['source'])
            for k,v in obj.items():
                if isinstance(v,str) and HEX.fullmatch(v) and ('/' in k or '\\' in k or Path(k).suffix in ('.py','.json','.pt','.npz','.npy','.zip')):
                    candidate=Path(k)
                    paths=[candidate] if candidate.is_absolute() else [context/candidate,package/candidate,workspace/candidate]
                    existing=[p for p in paths if p.is_file()]
                    yield existing[0] if existing else paths[0],v
                else:yield from references(v,context)
        elif isinstance(obj,list):
            for v in obj:yield from references(v,context)

    def flags(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k=='passed':yield v is True
                yield from flags(v)
        elif isinstance(obj,list):
            for v in obj:yield from flags(v)

    def violations(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k=='violations' and v:yield v
                yield from violations(v)
        elif isinstance(obj,list):
            for v in obj:yield from violations(v)

    def training_pairs(root, name):
        common=['history_initial','decoder_initial','normalization','inputs','targets','batches']
        count=0
        if name.startswith(('policy_finetune','mte_finetune')):
            for f in (root/'formal').glob('*/seed*/ground_*/training_identity.json'):
                suite=f.parents[2].name
                arm=f.parent.name
                if suite=='visual':arm=arm.replace('_trainable','_frozen')
                g=root/'parity'/suite/f.parents[1].name/arm/'training_identity.json'
                a,b=load(f),load(g)
                keys=common+(['total_history_parameters','decoder_parameters'] if suite=='visual' else ['history_parameters','decoder_parameters','fit_ids','validation_ids'])
                checked(all(a[k]==b[k] for k in keys),name,'training_pair',str(f))
                count+=1
            checked(count==COUNTS[name][0],name,'training_pair_count',count)
        elif name=='visual_supervision_inventory_2026_09_19':
            for seed,budget,composition in itertools.product(range(908711,908716),(1,2,4,8),('solo','aux')):
                identities=[]
                for init,mode in [('pretrained','frozen'),('pretrained','trainable'),('random','frozen'),('random','trainable')]:
                    phase='fair_parity' if (init,mode)==('pretrained','frozen') else 'formal'
                    identities.append(load(root/phase/f'seed{seed}'/f'b{budget}'/f'ground_mif_{composition}_{init}_{mode}'/'training_identity.json'))
                keys=[k for k in common if k!='history_initial']+['total_history_parameters','decoder_parameters']
                checked(all(all(r[k]==identities[0][k] for r in identities) for k in keys),name,'factorial_match',[seed,budget,composition])
                checked(identities[0]['history_initial']==identities[1]['history_initial'] and identities[2]['history_initial']==identities[3]['history_initial'] and identities[0]['history_initial']!=identities[2]['history_initial'],name,'factorial_initialization',[seed,budget,composition])
                count+=1
        elif name=='visual_branch_attribution_2026_09_20':
            for seed in range(908711,908716):
                sources=[]
                for family in ('simple','graph','tree','mif','laom'):
                    identities=[]
                    for mode in ('frozen','baseonly','auxonly','trainable'):
                        phase='parity' if mode in ('frozen','trainable') else 'formal'
                        a=load(root/phase/'visual'/f'seed{seed}'/f'ground_{family}_aux_pretrained_{mode}'/'training_identity.json')
                        for branch in ('base','aux'):
                            trainable=mode in (branch+'only','trainable')
                            checked(a[branch+'_trainable']==trainable and (a['branch_initial'][branch]!=a['branch_final'][branch])==trainable,name,'selective_branch_update',[seed,family,mode,branch])
                        identities.append(a)
                    keys=common+['total_history_parameters','decoder_parameters','base_parameters','aux_parameters']
                    checked(all(all(r[k]==identities[0][k] for r in identities) for k in keys),name,'four_cell_match',[seed,family])
                    count+=1;sources.append(identities[0])
                keys=['total_history_parameters','decoder_parameters','base_parameters','aux_parameters','decoder_initial','inputs','targets','batches']
                checked(all(all(r[k]==sources[0][k] for r in sources) for k in keys) and all(r['branch_initial']['base']==sources[0]['branch_initial']['base'] for r in sources),name,'cross_source_capacity_match',seed)
        return count

    def recompute_deltas(name,c,rows):
        # Independent aggregation from the archived per-controller records.
        seeds=sorted({r['seed'] for r in rows}) if rows and 'seed' in rows[0] else []
        if name.startswith(('policy_finetune','mte_finetune')):
            subset=sorted([r for r in rows if r['suite']==c['suite'] and r['arm']==c['arm']],key=lambda r:r['seed'])
            return [r['finetuned']-r['frozen'] for r in subset]
        if name=='visual_supervision_inventory_2026_09_19':
            v={(r['seed'],r['budget'],r['arm']):r['mean_return'] for r in rows}
            return [statistics.mean([sum(w*v[(s,b,a)] for a,w in c['terms']) for b in (1,2,4)]) for s in seeds]
        if name=='visual_low_label_2026_09_19_r1':
            v={(r['seed'],r['budget'],r['arm']):r['mean_return'] for r in rows}
            return [statistics.mean([v[(s,b,c['model'])]-v[(s,b,c['control'])] for b in (1,2,4)]) for s in seeds]
        if name=='mpe_matching_input_control_2026_09_19':
            v={(r['seed'],r['method']):r['mean_return'] for r in rows}
            return [v[(s,c['plus'])]-v[(s,c['minus'])] for s in seeds]
        if name=='mamujoco_route_completion_2026_09_19':
            arm=c['model']+'_matched'
            return [statistics.mean(r['mean_return'] for r in rows if r['seed']==s and r['arm']==arm)-statistics.mean(r['mean_return'] for r in rows if r['seed']==s and r['arm']==c['control']) for s in seeds]
        if name=='mamujoco_mif_visibility_2026_09_19':
            v={(s,a):statistics.mean(r['mean_return'] for r in rows if r['seed']==s and r['arm']==a) for s in seeds for a in ('masked_mobius','masked_raw','visible_mobius','visible_raw')}
            masked=[v[(s,'masked_mobius')]-v[(s,'masked_raw')] for s in seeds]
            visible=[v[(s,'visible_mobius')]-v[(s,'visible_raw')] for s in seeds]
            return masked if c['name']=='masked' else visible if c['name']=='visible' else [a-b for a,b in zip(masked,visible)]
        if name=='visual_branch_attribution_2026_09_20':
            v={(r['seed'],r['family'],r['mode']):r['mean_return'] for r in rows}
            family=c['family'];label=c['name']
            if label.startswith('minus_laom_'):
                mode=label.removeprefix('minus_laom_');return [v[(s,family,mode)]-v[(s,'laom',mode)] for s in seeds]
            terms={'base_update':{'baseonly':1,'frozen':-1},'aux_update':{'auxonly':1,'frozen':-1},'aux_update_given_base':{'trainable':1,'baseonly':-1},'base_update_given_aux':{'trainable':1,'auxonly':-1},'interaction':{'trainable':1,'baseonly':-1,'auxonly':-1,'frozen':1}}[label]
            return [sum(w*v[(s,family,m)] for m,w in terms.items()) for s in seeds]
        return None

    for entry in catalogue['experiments']:
        name=entry['id']; root=package/'outputs'/name; start=time.time()
        status=load(root/'status.json'); summary=load(root/'summary.json')
        checked(status['status']=='complete',name,'status',status.get('status'))
        rows=summary.get('records',summary.get('rows',[]))
        item=dict(id=name,status=status['status'],summary_sha256=digest(root/'summary.json'),
                  declared_hash_references=0,access_audits=0,result_references=0,mean_values_checked=0,
                  qualification_files=[],exact_statistic_checks=0)
        if name in COUNTS:
            n,keys=COUNTS[name]
            identities=[tuple(row.get(k) for k in keys) for row in rows]
            checked(len(rows)==len(set(identities))==n,name,'identities',dict(rows=len(rows),unique=len(set(identities)),expected=n))
            item['identity_check']=dict(rows=len(rows),unique=len(set(identities)),fields=keys,
                paired_frozen_and_adapt=name.startswith(('policy_finetune','mte_finetune')))
        if (root/'protocol.json').exists():item['protocol_sha256']=digest(root/'protocol.json')
        for key in ('source_and_weight_checks','weights_unchanged','code_unchanged','source_values_verified'):
            if key in summary:checked(summary[key] is True,name,key,summary[key])
        for path in sorted(root.rglob('*.json')):
            if any('interruption_' in part for part in path.parts):continue
            filename=path.name
            is_hash=(filename in ('manifest.json','source_manifest.json','implementation_sources.json','source_checkpoint_manifest.json','complete.json') or 'freeze' in filename)
            is_access='access' in filename
            is_qualification=filename in ('qualification.json','pairing_checks.json','factorial_checks.json')
            if not (is_hash or is_access or is_qualification):continue
            obj=load(path)
            if is_hash:
                if name=='mpe_partner_shift_2026_09_18_r1' and filename=='source_manifest.json':
                    # The original writer indexes decoder .pt hashes by its result .json path.
                    obj={**obj,'weights':{str(Path(k).with_suffix('.pt')):v for k,v in obj['weights'].items()}}
                for f,expected in references(obj,path.parent):
                    item['declared_hash_references']+=1
                    if not f.is_file():checked(False,name,'missing_hash_target',str(f));continue
                    actual=digest(f)
                    checked(actual==expected,name,'hash_mismatch',dict(path=str(f),expected=expected,actual=actual,manifest=str(path)))
                if filename=='manifest.json' and isinstance(obj,dict) and 'config_hash' in obj:
                    checked(digest(root/'protocol.json')==obj['config_hash'],name,'protocol_hash','manifest config_hash')
                for flag in flags(obj):checked(flag,name,'manifest_parity_flag',str(path))
            if is_access:
                item['access_audits']+=1
                checked(not list(violations(obj)),name,'access_violation',str(path))
            if is_qualification:
                item['qualification_files'].append(dict(path=str(path.relative_to(root)),sha256=digest(path)))
                for flag in flags(obj):checked(flag,name,'qualification',str(path))
        for row in rows:
            if not isinstance(row,dict):continue
            result_pairs=[(row.get('source'),row.get('mean_return')),
                          (row.get('frozen_source'),row.get('frozen'))]
            if name.startswith(('policy_finetune','mte_finetune')):
                result_pairs.append((str(root/'formal'/row['suite']/f"seed{row['seed']}"/f"evaluate_{row['arm']}"/'result.json'),row['finetuned']))
                checked(abs(row['delta']-(row['finetuned']-row['frozen']))<1e-8,name,'adapt_delta',str(row['arm']))
            for source,expected in result_pairs:
                if not isinstance(source,str):continue
                f=Path(source);checked(f.is_file(),name,'missing_result',str(f))
                if not f.is_file():continue
                r=load(f);item['result_references']+=1
                actual=r.get('mean_return',r.get('evaluation',{}).get('mean_return'))
                if isinstance(r.get('rows'),list) and len(r['rows'])==30 and all('return_value' in q for q in r['rows']):
                    # Historic B8 results store the all-30 mean. Follow-ups aggregate
                    # the predeclared 27 primary conditions from those SAME raw rows.
                    checked([q['seed'] for q in r['rows']]==list(range(908601,908631)),name,'visual_episode_ids',str(f))
                    primary=[q['return_value'] for q in r['rows'] if 908604<=q['seed']<=908630]
                    checked(len(primary)==27,name,'visual_primary_episodes',str(f))
                    actual=statistics.mean(primary)
                elif isinstance(r.get('episode_returns'),list):
                    value=statistics.mean(r['episode_returns'])
                    if actual is not None:checked(abs(value-actual)<1e-8,name,'episode_mean',str(f))
                    actual=value
                if expected is not None:
                    checked(actual is not None,name,'unrecognized_result_mean',str(f))
                    if actual is not None:
                        item['mean_values_checked']+=1
                        checked(abs(expected-actual)<1e-8,name,'mean_mismatch',dict(source=str(f),summary=expected,original=actual))
        item['training_identity_groups_rechecked']=training_pairs(root,name)
        groups={}
        for c in summary.get('contrasts',[]):
            d=c.get('seed_deltas',c.get('seed_differences'))
            if not d or 'exact_p' not in c:continue
            independent=recompute_deltas(name,c,rows)
            if independent is not None:checked(len(independent)==len(d) and max(abs(a-b) for a,b in zip(independent,d))<1e-8,name,'seed_delta_reaggregation',c.get('name',c.get('arm',c.get('model','contrast'))))
            label=c.get('name',c.get('arm',c.get('model','contrast')))
            mean=statistics.mean(d)
            declared=c.get('mean_delta',c.get('mean'))
            exact=sum(abs(statistics.mean([sign*v for sign,v in zip(signs,d)]))>=abs(mean)-1e-12
                      for signs in itertools.product((-1,1),repeat=len(d)))/(2**len(d))
            checked(len(d)==5,name,'replication_unit',label)
            checked(abs(mean-declared)<1e-8 and abs(exact-c['exact_p'])<1e-12,name,'exact_statistics',label)
            if 'positive_seeds' in c:
                checked(sum(v>0 for v in d)==c['positive_seeds'],name,'positive_seed_count',label)
            interval=c.get('t95',c.get('ci95'))
            if interval:
                from scipy.stats import t
                half=t.ppf(.975,len(d)-1)*statistics.stdev(d)/(len(d)**.5)
                checked(max(abs(interval[0]-(mean-half)),abs(interval[1]-(mean+half)))<1e-4,name,'t95',label)
            item['exact_statistic_checks']+=1
            family=c.get('test_family',c.get('composition',c.get('suite',c.get('family','all'))))
            groups.setdefault(family,[]).append((exact,c.get('holm_p'),label))
        for group,contrasts in groups.items():
            order=sorted(contrasts,key=lambda x:x[0]);running=0
            for i,(p,adjusted,label) in enumerate(order):
                running=min(1.,max(running,(len(order)-i)*p))
                if adjusted is not None:checked(abs(running-adjusted)<1e-12,name,'holm_family',dict(family=group,contrast=label,recomputed=running,declared=adjusted))
        item['failures']=[f for f in failures if f['run']==name]
        item['seconds']=time.time()-start;reports.append(item)
        if item['failures']:print(json.dumps(item['failures'][:5]),flush=True)
        print(json.dumps(dict(run=name,hashes=item['declared_hash_references'],audits=item['access_audits'],failures_so_far=len(failures))),flush=True)
    return dict(status='pass' if not failures else 'requires_investigation',runs=reports,
                unique_files_hashed=len(hash_cache),failures=failures,
                scope='Declared archive hash references, access audits, identity matrices, referenced return means and declared exact five-seed statistics. Does not retrain or certify every scientific assumption.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--catalogue',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.package,read(args.catalogue))
    with args.report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=result['status'],files=result['unique_files_hashed'],failures=len(result['failures']))))
