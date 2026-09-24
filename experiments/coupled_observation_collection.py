"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import hashlib

import json

from pathlib import Path

P=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=True,indent=2),encoding='utf-8');q.replace(p)

def unchanged(m):
    for p,h in m.items():assert sha(p)==h,('source drift',p)

def identity():
    import psutil
    p=psutil.Process();return dict(pid=p.pid,created=p.create_time(),command=p.cmdline())

def teachers(c):return P/c['teacher_root']

def teacher_path(c,s):return teachers(c)/f'seed{s}'/'teacher_final.zip'

def curator(c,s):
    from stable_baselines3 import SAC
    path=teacher_path(c,s);f=read(path.parent/'teacher_freeze.json')
    assert sha(path)==f['checkpoint_sha256'] and f['steps']==100000 and f['updates']==90000
    return SAC.load(path,device='cpu')

def collect(c,s,out,progress):
    import numpy as np
    from PIL import Image
    from environments.mamujoco import make_env,state_vector,_step
    from environments.teacher import global_to_local_action
    from visual.coupled_cheetah_sensor_dev import CoupledCheetahSensor
    model=curator(c,s); seeds=c['fit_seeds']+c['validation_seeds'];rows=[]
    states=[];pixels=[];target=[];trace_dir=out/'curator_traces';trace_dir.mkdir()
    for ix,seed in enumerate(seeds):
        env=make_env(seed,'CoupledHalfCheetah','1p1');sensor=None
        try:
            sensor=CoupledCheetahSensor(env,mode='clean',seed=seed+1000)
            rgb=[];ss=[];aa=[];rr=[];qpos=[];qvel=[];parts=[];times=[]
            native=env.single_agent_env.unwrapped
            for t in range(c['steps']+1):
                before=(native.data.qpos.copy(),native.data.qvel.copy(),float(native.data.time))
                frame=sensor.pixels()
                assert np.array_equal(before[0],native.data.qpos) and np.array_equal(before[1],native.data.qvel) and before[2]==float(native.data.time)
                rgb.append(frame);ss.append(state_vector(env));qpos.append(before[0]);qvel.append(before[1]);times.append(before[2])
                if t==c['steps']:break
                glob,_=model.predict(ss[-1],deterministic=True);action=global_to_local_action(env,glob)
                result=_step(env,action);info=result[4]['agent_0'];reward=result[1]['agent_0']
                assert abs(reward-(info['reward_run']-(info['reward_ctrl1']+info['reward_ctrl2'])/2))<1e-10
                aa.append(action.copy());rr.append(reward);parts.append([info['reward_run'],info['reward_ctrl1'],info['reward_ctrl2']])
                assert env.agents, 'Unexpected termination before fixed episode length'
                sensor.advance()
            rgb=np.stack(rgb);ss=np.stack(ss);aa=np.stack(aa)
            pixels.append(rgb);states.append(ss);target.append(aa[:,0].copy())
            trace=trace_dir/f'{seed}.npz'
            np.savez_compressed(trace,qpos=qpos,qvel=qvel,actions=aa,rewards=rr,reward_parts=parts,times=times)
            if ix==0:
                montage=np.concatenate([np.concatenate([rgb[t,0],rgb[t,1]],axis=1) for t in [0,50,100,199]],axis=0)
                Image.fromarray(montage).save(out/'clean_preview.png')
            rows.append(dict(seed=seed,split='fit' if ix<len(c['fit_seeds']) else 'validation',steps=c['steps'],return_sum=float(np.sum(rr)),trace=str(trace.resolve()),sha256=sha(trace)))
            progress(episode=ix+1,total=len(seeds),seed=seed)
        finally:
            if sensor is not None:sensor.close()
            env.close()
    np.save(out/'rgb.npy',np.stack(pixels),allow_pickle=False)
    np.save(out/'state.npy',np.stack(states),allow_pickle=False)
    np.save(out/'target_actions.npy',np.stack(target),allow_pickle=False)
    return dict(teacher=s,rows=rows,teacher_qualified=read(teachers(c)/f'seed{s}'/'result.json')['teacher_qualified'],
                frame_shape=list(np.stack(pixels).shape),labels_shape=list(np.stack(target).shape),
                native_physics_unchanged=True,teacher_updates=0,learner_inputs='separate rgb/state/target-action files; joint labels only in curator audit traces',
                files={str((out/x).resolve()):sha(out/x) for x in ['rgb.npy','state.npy','target_actions.npy']})
