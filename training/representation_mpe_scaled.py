"""Dimension-only entity/table builder; no PC model redefinition."""
import numpy as np

def endpoint_view(positions, actual, reference, masks):
    """Actual learned endpoints, with unchanged landmark positions, not carriers."""
    e,t,h,c,d=actual.shape
    agents=d//2; contexts=2**(agents-1)
    assert agents in (4,6,8) and positions.shape==(e,26,4*agents)
    assert actual.shape==reference.shape and (t,h,c,d)==(22,3,contexts,2*agents)
    values=[]
    entities=[]
    for agent_endpoints in (actual,reference):
        full=np.zeros((e,t,h,c,4*agents),np.float32)
        full[...,:2*agents]=agent_endpoints
        full[...,2*agents:]=positions[:,1:23,None,None,2*agents:]
        table=np.zeros((e,t,h,c,2*agents,4),np.float32)
        table[...,:2]=full.reshape(e,t,h,c,2*agents,2)
        table[...,:agents,2]=1.;table[...,agents:,3]=1.
        values.append(full);entities.append(table)
    a,b=values
    adjacency=np.eye(2*agents,dtype=np.float32)
    adjacency[:agents,:]=1.;adjacency[:, :agents]=1.
    return {'obs':positions[:,1:24], 'actions':np.zeros((e,t,2),np.float32),
            'cf_root':a[:,:,0,0], 'cf_ego_null':b[:,:,0,0],
            'cf_other_null':a[:,:,0,-1], 'cf_both_null':b[:,:,0,-1],
            'cf_context_root':a[:,:,0], 'cf_context_ego_null':b[:,:,0],
            'cf_context_masks':masks, 'effect_horizons':np.arange(1,4),
            'cf_horizon_entity_root':entities[0], 'cf_horizon_entity_ego_null':entities[1],
            'mif_entity_adjacency':adjacency}

