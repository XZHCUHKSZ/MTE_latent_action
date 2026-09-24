"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

def feature_slice(z,kind):
    if kind=='base':return z[...,:16]
    if kind=='auxiliary':
        if z.shape[-1]!=32:raise ValueError('Expected [Base16, Auxiliary16]')
        return z[...,16:]
    if kind=='combined':return z
    raise ValueError('Use base, auxiliary or combined')
