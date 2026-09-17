"""X6. Retain readable learned partner interactions, not physical causal truth."""
from evaluation.probes import reader,transform
from mte.coordinate_controls import factory,forward
from closed_loop_lam_v1.unified_models import lattice_operators
import numpy as np

def readout(z_train,z_dev,y_train,y_dev,kind,masks):
    """Per-partner readout extracted from agent_resolved_probe.main's inner loop.

    Coefficients have [row,horizon,subset,entity,coordinate]; observed changes
    have [row,agent,XY_or_two_angles]. Use the saved train/dev selections and
    train-only canonical target scaling; do not fit a scale on dev targets.
    """
    pred=reader(z_train.astype(float),y_train.reshape(len(y_train),-1).astype(float),z_dev.astype(float)).reshape(y_dev.shape)
    mean=y_train.mean(0);rows=[]
    if kind not in {'coefficient','observed_change'}:raise ValueError(kind)
    for k in range(len(masks) if kind=='coefficient' else 4):
        if kind=='coefficient':
            p,q,mu=pred[:,:,k],y_dev[:,:,k],mean[:,k]
            label=','.join(str(i) for i in np.flatnonzero(masks[k])) or 'empty'
        else:p,q,mu=pred[:,k],y_dev[:,k],mean[k];label=str(k)
        mse=float(np.mean((p-q)**2));den=float(np.mean((q-mu)**2))
        rows.append(dict(kind=kind,group=label,mse=mse,training_mean_mse=den,skill=1-mse/den if den>1e-12 else None,latent_dimension=z_train.shape[1]))
    return rows

# reader is the original fixed-ridge diagnostic; transform is the original
# learned-coefficient construction. Use representation and history features
# separately. Coordinate pilot: two development seeds, 400 updates, no rollout.
