"""Post-grounding partner-policy transforms; existing learned classes unchanged."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from closed_loop_lam_v1.common import RecurrentPolicy, ActionDecoder
from training.composition import FrozenPair
from training.grounding_mpe import GroundedPolicy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_grounded(path, method, budget):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['method']==method and ck['budget']==budget
    state=ck['history']
    def recurrent(prefix=''):
        sub={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        net=RecurrentPolicy(sub['gru.weight_ih_l0'].shape[1], sub['head.2.weight'].shape[0],
                            sub['gru.weight_hh_l0'].shape[1])
        net.load_state_dict(sub,strict=True)
        return net.eval()
    net=FrozenPair(recurrent('anchor.'),recurrent('aux.')) if '+' in method else recurrent()
    # Load the entire saved state too, ensuring no composition key is omitted.
    net.load_state_dict(state,strict=True)
    decoder=None
    if ck['decoder'] is not None:
        dec=ck['decoder'];decoder=ActionDecoder(dec['net.0.weight'].shape[1],
                dec['net.4.weight'].shape[0],dec['net.0.weight'].shape[0])
        decoder.load_state_dict(dec,strict=True);decoder.eval()
    for module in [net,decoder]:
        if module is not None:
            for parameter in module.parameters():parameter.requires_grad_(False)
    return GroundedPolicy(net,ck['mean'],ck['std'],decoder)


def change_partners(joint,condition):
    result=joint.copy()
    if condition=='slow_0.5':result[1:]*=.5
    elif condition in ('rotate_plus30','rotate_minus30'):
        angle=np.pi/6*(1 if condition=='rotate_plus30' else -1)
        matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        result[1:]=result[1:]@matrix.T
    elif condition!='identity':raise ValueError(condition)
    assert np.array_equal(result[0],joint[0])
    assert np.max(np.abs(result))<=1.00001
    return result


def evaluate(policy,seeds,condition,anchor=None):
    from environments.mpe import make_env,teacher_forces,team_reward,collision_count,coverage_distance
    from environments.native_particle import force_to_direction_id
    rows=[]
    for seed in seeds:
        env=make_env(int(seed),4,23);rng=np.random.default_rng(int(seed)+901)
        random_rng=np.random.default_rng(int(seed)+1901)
        hidden=None;reward=0.;cols=0.
        try:
            for t in range(23):
                position=np.concatenate([env.all_agent_pos().reshape(-1),env.all_landmark_pos().reshape(-1)]).astype(np.float32)
                joint=teacher_forces(env,'high',rng,None)
                if anchor=='teacher':action=joint[0].copy()
                elif anchor=='random':action=random_rng.uniform(-1,1,2).astype(np.float32)
                else:action,hidden=policy.act(position,hidden)
                if t==0:action=np.zeros(2,np.float32)
                joint=change_partners(joint,condition);joint[0]=action
                env.step_with_forces(joint,np.asarray([force_to_direction_id(a) for a in joint],np.int64))
                reward+=team_reward(env);cols+=collision_count(env)
            rows.append(dict(episode_seed=int(seed),return_=reward,collisions=cols,coverage=coverage_distance(env)))
        finally:env.env.close()
    return dict(condition=condition,episodes=rows,mean_return=float(np.mean([r['return_'] for r in rows])),
        mean_collisions=float(np.mean([r['collisions'] for r in rows])),
        mean_coverage=float(np.mean([r['coverage'] for r in rows])))
