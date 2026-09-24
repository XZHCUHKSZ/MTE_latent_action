import numpy as np

# Final implementation source: mte_observation_only_2026_09_09/grounding.py:175

def evaluate(policy,seeds,anchor=None):
    # Only this post-training stage imports/executes the simulator.
    from environments.mpe import make_env, teacher_forces, team_reward, collision_count, coverage_distance
    from environments.native_particle import force_to_direction_id
    returns=[];collisions=[];coverage=[]
    for seed in seeds:
        env=make_env(int(seed),4,23);rng=np.random.default_rng(int(seed)+901)
        random_rng=np.random.default_rng(int(seed)+1901)
        hidden=None;reward=0.;cols=0.
        try:
            for t in range(23):
                position=np.concatenate([env.all_agent_pos().reshape(-1),env.all_landmark_pos().reshape(-1)]).astype(np.float32)
                joint=teacher_forces(env,'high',rng,None)
                if anchor=='teacher': action=joint[0].copy()
                elif anchor=='random': action=random_rng.uniform(-1,1,2).astype(np.float32)
                else: action,hidden=policy.act(position,hidden)
                if t==0: action=np.zeros(2,np.float32)
                joint[0]=action
                ids=np.asarray([force_to_direction_id(a) for a in joint],np.int64)
                env.step_with_forces(joint,ids);reward+=team_reward(env);cols+=collision_count(env)
            returns.append(reward);collisions.append(cols);coverage.append(coverage_distance(env))
        finally: env.env.close()
    return dict(episode_seeds=list(seeds),episode_returns=returns,episode_collisions=collisions,
        episode_final_coverage=coverage,mean_return=float(np.mean(returns)),
        mean_collisions=float(np.mean(collisions)),mean_final_coverage_distance=float(np.mean(coverage)),
        evaluation_steps=23,common_warmup_action=[0.,0.],partner_policy='high quality, dynamic assignment',
        evidence='simulator development closed loop; not original 25-step score')
