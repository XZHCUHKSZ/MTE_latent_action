"""Scientific reporting and scheduling contracts; no performance assertions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from evaluation.temporal_scale_report import normalized_auc,paired_stats,holm
from experiments.mpe_temporal_scale import paths


def main():
    assert normalized_auc([2]*6,[8,16,32,64,128,256])==2
    # Linear values on evenly spaced log2 budgets have analytic mean 2.5.
    assert normalized_auc(range(6),[8,16,32,64,128,256])==2.5
    positive=paired_stats([1,2,3,4,5]);negative=paired_stats([-1,-2,-3,-4,-5])
    assert positive['mean']==-negative['mean']
    assert np.isclose(positive['p'],negative['p'])
    assert np.allclose(positive['ci95'],[-negative['ci95'][1],-negative['ci95'][0]])
    assert np.allclose(holm([.03,.001,.02]),[.04,.003,.04])
    assert paths(Path('run'),'ground',45,8)[2]!=paths(Path('run'),'ground',45,16)[2]
    p=json.loads((Path(__file__).resolve().parents[1]/'configs/mpe_temporal_full_scale.json').read_text())
    assert len(p['seeds'])*len(p['budgets'])*len(p['control_configs'])==480
    assert not set(p['seeds'])&{40,41}
    assert set(p['data_seed_map'].values())=={40,41,42,43,44}
    print('PASS: log-budget AUC, sign-symmetric paired statistics, Holm, disjoint budget writes, full matrix')


if __name__=='__main__':main()
