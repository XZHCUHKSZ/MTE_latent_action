"""X5. Test action information readable from frozen history features.

Representation pretraining does not read actions. These supervised diagnostic
probes are fitted AFTER freezing. Auxiliary-only excludes Base coordinates.
"""
from evaluation.probes import ridge_probe

def feature_slice(z,kind):
    if kind=='base':return z[...,:16]
    if kind=='auxiliary':
        if z.shape[-1]!=32:raise ValueError('Expected [Base16, Auxiliary16]')
        return z[...,16:]
    if kind=='combined':return z
    raise ValueError('Use base, auxiliary or combined')
