"""Scientific contracts for the evaluation-only native-effect diagnostic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from evaluation.effect_statistics import partition, target_edges, single_partner_interactions, fit_reader


def main():
    # Analytic assignment cube: F = 2*x_target + 3*x_partner + 5*x_target*x_partner.
    # Replaced agents have x=0. Thus g(empty)=7, g({partner})=2, c=-5.
    cube = np.array([2*(1-(i&1))+3*(1-((i>>1)&1))+5*(1-(i&1))*(1-((i>>1)&1)) for i in range(16)])
    edge = target_edges(cube[None, :, None])
    assert edge[0, 0, 0] == 7 and edge[0, 1, 0] == 2
    assert single_partner_interactions(edge)[0, 0, 0] == -5
    assert np.all(single_partner_interactions(edge)[0, 1:] == 0)
    ids = np.repeat(np.arange(12), 3)
    sp = partition(ids, range(6), range(6, 9), range(9, 12))
    try:
        partition(ids, range(7), range(6, 9), range(9, 12))
    except ValueError:
        pass
    else:
        raise AssertionError('Episode split overlap was accepted')
    rng = np.random.default_rng(17)
    x = rng.normal(size=(36, 3)); y = x @ rng.normal(size=(3, 2))
    pred, _ = fit_reader(x, y, sp, np.ones(2), [.01, .1])
    corrupted = y.copy(); corrupted[sp[2]] += 100
    pred2, _ = fit_reader(x, corrupted, sp, np.ones(2), [.01, .1])
    assert np.array_equal(pred, pred2), 'Test targets influenced fitted reader'
    print('PASS: cube sign, partner indexing, episode separation, test-label independence')
    from environments.effect_evaluation import EffectEnvironment
    env = EffectEnvironment('mpe', 97179999)
    try:
        snap = env.snapshot()
        zero = np.zeros((3, 4, 2), np.float32)
        pulse = zero.copy(); pulse[0, 0, 0] = .5
        a, _ = env.roll(snap, pulse); b, _ = env.roll(snap, zero)
        repeated, _ = env.roll(snap, pulse)
        assert np.array_equal(a, repeated), 'Non-reproducible physical replay'
        assert np.array_equal(a[0], b[0]), 'MPE integrator changed: reassess physical horizon protocol'
        assert np.max(np.abs(a[1]-b[1])) > 1e-4, 'Target action has no delayed physical effect'
        print('PASS: MPE pulse action leaves h1 positions unchanged and changes h2; repeated rollout exact')
    finally:
        env.close()


if __name__ == '__main__':
    main()
