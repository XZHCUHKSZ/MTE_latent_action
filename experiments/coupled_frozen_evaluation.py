"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import argparse,time,sys,subprocess,traceback

from experiments.coupled_observation_collection import P,read,save,sha,unchanged,identity,curator

def jobroot(root,s,seed):return root/f'teacher{s}'/f'seed{seed}'

def evaluate(c,s,seed,root,out,note):
    import numpy as np,torch
    from training.coupled_cheetah_frozen import Controller,arms
    from environments.mamujoco import make_env,state_vector,_step
    from environments.teacher import global_to_local_action
    from visual.coupled_cheetah_sensor_dev import CoupledCheetahSensor
    torch.set_num_threads(c['threads']);teacher=curator(c,s);jr=jobroot(root,s,seed);fr=read(root/'ground_freeze.json');unchanged(fr['files']);rows=[];start=time.time()
    for arm in arms():
        controller=Controller(jr/'pre',jr/'ground'/arm/'controller.pt')
        for ev in c['evaluation_seeds']:
            env=make_env(ev,'CoupledHalfCheetah','1p1');sensor=None;native=env.single_agent_env.unwrapped;controller.reset()
            fields={k:[] for k in ['states','actions','teacher_actions','rewards','reward_parts','qpos','qvel','times','terminated','truncated']}
            try:
                sensor=CoupledCheetahSensor(env,mode='clean',seed=ev+1000)
                fields['qpos'].append(native.data.qpos.copy());fields['qvel'].append(native.data.qvel.copy());fields['times'].append(float(native.data.time))
                for t in range(200):
                    st=state_vector(env);g,_=teacher.predict(st,deterministic=True);ta=global_to_local_action(env,g);a=ta.copy();before=(native.data.qpos.copy(),native.data.qvel.copy(),float(native.data.time));rgb=sensor.pixels()
                    assert np.array_equal(before[0],native.data.qpos) and np.array_equal(before[1],native.data.qvel) and before[2]==float(native.data.time)
                    a[0]=controller.act(rgb);rr=_step(env,a);info=rr[4]['agent_0'];rw=rr[1]['agent_0'];assert rw==rr[1]['agent_1'] and abs(rw-(info['reward_run']-(info['reward_ctrl1']+info['reward_ctrl2'])/2))<1e-10
                    for k,v in dict(states=st,actions=a.copy(),teacher_actions=ta,rewards=rw,reward_parts=[info['reward_run'],info['reward_ctrl1'],info['reward_ctrl2']],qpos=native.data.qpos.copy(),qvel=native.data.qvel.copy(),times=float(native.data.time),terminated=rr[2]['agent_0'],truncated=rr[3]['agent_0']).items():fields[k].append(v)
                    assert env.agents and not rr[2]['agent_0'] and not rr[3]['agent_0'];sensor.advance()
                path=out/f'{arm}_{ev}.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in fields.items()});rows.append(dict(teacher=s,training_seed=seed,arm=arm,seed=ev,steps=200,return_sum=float(np.sum(fields['rewards'])),trace=str(path.resolve()),sha256=sha(path)));note(stage='evaluate',completed=len(rows),total=len(arms())*len(c['evaluation_seeds']),arm=arm)
            finally:
                if sensor is not None:sensor.close()
                env.close()
    unchanged(fr['files']);return dict(rows=rows,started_unix=start,qualification=c['qualification'])
