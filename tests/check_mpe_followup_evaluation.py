"""Scientific interface checks for evaluation-only review follow-ups."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.mpe_physical_interaction_onset import second_difference
from evaluation.mpe_partner_shift import change_partners

# Known action interaction: additive terms cancel, cross term remains.
def f(a,b):return np.array([2*a+3*b+5*a*b],dtype=np.float32)
assert np.array_equal(second_difference(f(1,1),f(0,1),f(1,0),f(0,0)),[5.])
assert np.array_equal(second_difference(f(1,1),f(1,1),f(0,1),f(0,1)),[0.])
joint=np.array([[.2,-.3],[1,0],[0,1],[-.6,.8]],np.float32)
saved=joint.copy()
for condition in ('identity','slow_0.5','rotate_plus30','rotate_minus30'):
    changed=change_partners(joint,condition)
    assert np.array_equal(joint,saved), 'Input mutated'
    assert np.array_equal(changed[0],joint[0]), 'Target transformed with partners'
    ratio=.5 if condition=='slow_0.5' else 1.
    assert np.allclose(np.linalg.norm(changed[1:],axis=1),ratio*np.linalg.norm(joint[1:],axis=1))
assert np.allclose(change_partners(change_partners(joint,'rotate_plus30'),'rotate_minus30'),joint)
assert change_partners(joint,'rotate_plus30')[1,1]>.49
assert change_partners(joint,'rotate_minus30')[1,1]<-.49
print('PASS: physical difference sign/null control; target invariance; rotation sign/norm; input immutability.')
