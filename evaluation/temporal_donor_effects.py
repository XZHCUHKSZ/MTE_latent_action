"""Actual nearest-training-history donor, used only in frozen physical evaluation."""
import json
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from mte.frontends import ObservationReadout
from environments.effect_evaluation import EffectEnvironment
from evaluation.probes import ridge_probe
from evaluation.temporal_completion_probes import score
from experiments.mpe_temporal_stages import digest
from utils.access import read_rows
from utils.atomic import atomic_json


def donor_bank(n,p):
    rng=np.random.default_rng(p['seed']+11)
    for _ in range(p['readout_updates']):rng.integers(n*22,size=p['batch'])
    return rng.choice(n*22,min(1024,n*22),replace=False)


def evaluate(root,p,workspace,c,progress):
    frozen=root/'pretrain';before=json.loads((frozen/'complete.json').read_text())['checkpoints']
    bc=torch.load(frozen/'offset2/bridge/readout.pt',map_location='cpu',weights_only=False)
    fc=torch.load(frozen/'offset2/entity/frontend.pt',map_location='cpu',weights_only=False)
    entity=C.LAOMStateAdapter(2,16,64);entity.load_state_dict(fc['state_dict']);entity.eval()
    predictor=ObservationReadout(16,64,24);predictor.load_state_dict(bc['state_dict']);predictor.eval()
    with np.load(root/'data/train.npz') as f:x=f['positions']
    with np.load(frozen/'offset2/entity/codes.npz') as f:z=f['z'][:len(x)].reshape(-1,4,16)
    bank=donor_bank(len(x),p)
    hist=np.concatenate([x[:,:22],x[:,1:23]],-1).reshape(-1,32)
    ht=(hist-bc['history_mean'])/bc['history_std'];bh=ht[bank]
    # Verify exact reproduction of the archived training donor-bank procedure.
    with np.load(frozen/'offset2/bridge/endpoints.npz') as f:stored=f['donor_flat_ids']
    td=torch.cdist(torch.tensor(ht[:64]),torch.tensor(bh))
    td.masked_fill_(torch.tensor(np.arange(64)[:,None]//22==bank[None]//22),float('inf'))
    assert np.array_equal(bank[td.argmin(1).numpy()],stored[:64])
    meta=json.loads((root/'data/manifest.json').read_text())
    joint=read_rows(meta['source'],'joint_actions',meta['train_ids'])[:,1:23].reshape(-1,4,2)
    rows=[];records=[];count=c['donor_episodes'];rng=np.random.default_rng(918101+p['seed'])
    def code(now,future):
        aa=torch.tensor((now[:8].reshape(-1,2)-fc['mean'])/fc['std'])
        bb=torch.tensor((future[:8].reshape(-1,2)-fc['mean'])/fc['std'])
        with torch.no_grad():return entity(aa,bb)[1].numpy()
    def predict(h,zz):
        h=torch.tensor(h[None]);zz=torch.tensor((zz.reshape(1,64)-bc['zmean'])/bc['zstd'])
        with torch.no_grad():return predictor(h,zz).numpy().reshape(3,8)*bc['delta_std']+bc['delta_mean']
    for ep in range(count):
        env=EffectEnvironment('mpe',c['donor_seed_start']+ep)
        try:
            previous=env.observation()
            for t in range(15):
                current=env.observation();action=env.action()
                if t==0:action[0]=0
                if t in (2,6,10,14):
                    snap=env.snapshot();seq=np.repeat(action[None],3,0)
                    factual,_=env.roll(snap,seq);replay,_=env.roll(snap,seq)
                    assert np.array_equal(factual,replay)
                    h=(np.concatenate([previous,current])-bc['history_mean'])/bc['history_std']
                    di=bank[torch.cdist(torch.tensor(h[None]),torch.tensor(bh)).argmin().item()]
                    random_di=int(rng.choice(bank));actual=code(current,factual[1]);plus=predict(h,actual)
                    for kind,donor in [('nearest_history',di),('random_donor',random_di)]:
                        seqminus=seq.copy();seqminus[0,0]=joint[donor,0]
                        ref,_=env.roll(snap,seqminus);truth=factual[:,:8]-ref[:,:8]
                        assert np.array_equal(factual[0],ref[0])
                        minus=actual.copy();minus[0]=z[donor,0]
                        matched=plus-predict(h,minus);mismatch=plus-predict(h,z[donor])
                        restoredcode=code(current,ref[1])
                        rec=dict(episode=ep,t=t,kind=kind,donor=int(donor),truth=truth,
                            matched=matched,mismatched=mismatch,history=h,
                            target_pair=np.concatenate([actual[0],z[donor,0]]),
                            target_code_transport_gap=float(np.square(z[donor,0]-restoredcode[0]).mean()),
                            partner_code_change=float(np.square(actual[1:]-restoredcode[1:]).mean()),
                            history_distance=float(np.linalg.norm(h-ht[donor])))
                        records.append(rec)
                    env.restore(snap)
                previous=current;env.step(action)
        finally:env.close()
        progress('donor_effects',episode=ep+1,episodes=count)
    for kind in ('nearest_history','random_donor'):
        rr=[r for r in records if r['kind']==kind]
        truth=np.stack([r['truth'] for r in rr]);scale=bc['delta_std']
        for mode in ('matched','mismatched','zero'):
            pred=np.zeros_like(truth) if mode=='zero' else np.stack([r[mode] for r in rr])
            for h in (1,2,3):
                rows.append(dict(kind=kind,mode=mode,horizon=h,
                    mse=float(np.square(pred[:,h-1]-truth[:,h-1]).mean()),
                    standardized_mse=float(np.square((pred[:,h-1]-truth[:,h-1])/scale[h-1]).mean()),
                    true_effect_rms=float(np.sqrt(np.square(truth[:,h-1]).mean()))))
    rr=[r for r in records if r['kind']=='nearest_history'];eps=np.array([r['episode'] for r in rr])
    truth=np.stack([r['truth'][1] for r in rr]);history=np.stack([r['history'] for r in rr])
    fit=eps<count//2;val=(eps>=count//2)&(eps<3*count//4);test=eps>=3*count//4
    readers=[]
    for name,features in [('history_only',history),
        ('history_target_slot_pair',np.concatenate([history,np.stack([r['target_pair'] for r in rr])],-1)),
        ('history_mte_h2',np.concatenate([history,np.stack([r['matched'][1] for r in rr])],-1))]:
        pred,diag,_=ridge_probe(features[fit].astype(float),truth[fit],features[val].astype(float),truth[val],
            features[test].astype(float),[.0001,.001,.01,.1,1.,10.])
        readers.append(dict(method=name,**score(pred,truth[test],truth[fit].mean(0)),reader=diag))
    arrays={k:np.stack([r[k] for r in records]) for k in ('truth','matched','mismatched','history','target_pair')}
    np.savez_compressed(root/'donor_predictions_evaluation_only.npz',**arrays,
                        episode=np.array([r['episode'] for r in records]),donor=np.array([r['donor'] for r in records]))
    assert all(digest(frozen/f)==v for f,v in before.items())
    atomic_json(root/'donor_effects.json',dict(seed=p['seed'],episodes=count,rows=rows,readers=readers,
        donor_bank_reproduced=True,replay_exact=True,training_updates=0,weights_unchanged=True,
        transport_diagnostics={key:float(np.mean([r[key] for r in rr])) for key in
            ('target_code_transport_gap','partner_code_change','history_distance')},
        protocol='Nearest original training-history bank; target donor action replaces only first action; same factual continuation. Random donor has its own paired physical truth.',
        scope='Evaluation-only simulator/action access. Uncalibrated effect errors separate from calibrated readers. h1 true-zero negative control retained. Reader widths differ; no claim of resource matching.',
        reference='Original training donor rule tested on fresh evaluation states; donor codes are not assumed to identify native actions.'))
