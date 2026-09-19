"""Measured encoder resource scaling; never presented as trained control evidence."""
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import torch
from closed_loop_lam_v1.unified_models import MobiusSimple,MobiusGraphEdgeCARA
from environments.mpe import make_env,teacher_assignment,teacher_forces
from environments.native_particle import force_to_direction_id
from utils.atomic import atomic_json
P=Path(__file__).resolve().parents[1]

def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()

def main():
    cfg=P/'configs/mte_agent_resource_audit.json';p=json.loads(cfg.read_text());out=P/'outputs'/p['name'];out.mkdir(exist_ok=False)
    torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    files=[Path(__file__),cfg,P/'closed_loop_lam_v1/unified_models.py',P/'environments/mpe.py',P/'environments/native_particle.py']
    sources={str(f):sha(f) for f in files};atomic_json(out/'protocol.json',p);atomic_json(out/'source_manifest.json',sources)
    env_checks=[];rows=[]
    for agents in p['agent_counts']:
        env=make_env(p['seed'],num_agents=agents,max_cycles=25)
        assignment=teacher_assignment(env);rng=np.random.default_rng(p['seed'])
        before=env.all_agent_pos().copy()
        for _ in range(4):
            forces=teacher_forces(env,'high',rng,assignment)
            ids=np.array([force_to_direction_id(f) for f in forces],dtype=np.int64)
            env.step_with_forces(forces,ids)
        after=env.all_agent_pos();assert after.shape==(agents,2) and np.isfinite(after).all()
        env_checks.append(dict(agents=agents,landmarks=len(env.all_landmark_pos()),position_shape=list(after.shape),motion_l2=float(np.linalg.norm(after-before)),passed=True,scope='four teacher-forced steps; no trained policy'))
        env.env.close()
        partners=agents-1;k=2**partners
        masks=torch.tensor([[(i>>j)&1 for j in range(partners)] for i in range(k)],dtype=torch.float32)
        for width_kind in p['input_widths']:
            width=8 if width_kind=='fixed8' else 2*agents
            for name,cls in [('Simple',MobiusSimple),('Graph',MobiusGraphEdgeCARA)]:
                torch.manual_seed(p['seed']);start=time.perf_counter()
                model=cls(width,masks,p['latent_dim'],p['hidden'],**({'layers':p['layers']} if name=='Graph' else {}))
                construction=time.perf_counter()-start
                edges=torch.randn(p['batch'],k,width);forward=[];backward=[]
                for i in range(p['warmups']+p['repeats']):
                    model.zero_grad(set_to_none=True);t=time.perf_counter();result=model(edges)
                    loss=(result[-1]-edges).square().mean();t2=time.perf_counter();loss.backward();t3=time.perf_counter()
                    assert torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
                    if i>=p['warmups']:forward.append((t2-t)*1000);backward.append((t3-t2)*1000)
                row=dict(agents=agents,partners=partners,contexts=k,model=name,width_kind=width_kind,input_width=width,
                    parameters=sum(q.numel() for q in model.parameters()),parameter_bytes=sum(q.numel()*q.element_size() for q in model.parameters()),
                    registered_buffer_bytes=sum(q.numel()*q.element_size() for q in model.buffers()),input_bytes=edges.numel()*edges.element_size(),
                    constructor_seconds=construction,forward_ms=forward,backward_ms=backward,median_forward_ms=statistics.median(forward),median_backward_ms=statistics.median(backward))
                rows.append(row);atomic_json(out/'progress.json',dict(completed=len(rows),total=16,last=row))
                del result,loss,model,edges;gc.collect()
    analytical=[dict(partners=pv,contexts=2**pv,one_dense_float32_matrix_bytes=4*(2**pv)**2,mobius_plus_zeta_bytes=8*(2**pv)**2) for pv in p['analytical_partner_counts']]
    assert all(sha(Path(f))==h for f,h in sources.items())
    result=dict(rows=rows,environment_interface_checks=env_checks,analytical=analytical,source_checks=True,cpu=platform.processor(),torch=torch.__version__,device='cpu',threads=1,concurrency_note='Other GPU experiment may contend for host CPU; timing descriptive, not a hardware-normalized leaderboard. Model initialization and matrix factorization measured separately; end-to-end table construction/history/decoder/simulator not timed.')
    atomic_json(out/'summary.json',result)
    lines=['# Agent数量与原编码器资源审计','','原Simple/Graph类未修改；CPU单线程、batch64、latent16、hidden128。随机表输入，非学习曲线。4/6/8/10 agent原环境各运行4步教师动作，只验证接口。','固定8输出坐标与随agent增长的2A坐标分别测量；计时为7次重复中位数，构造矩阵另计。并发GPU任务可能争用CPU，时间是当前主机描述性测量。','','|agent|模型|输入宽度|子集数|参数|buffer MiB|前向 ms|反向 ms|','|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['agents']}|{r['model']}|{r['input_width']} ({r['width_kind']})|{r['contexts']}|{r['parameters']}|{r['registered_buffer_bytes']/2**20:.3f}|{r['median_forward_ms']:.3f}|{r['median_backward_ms']:.3f}|")
    lines+=['','完整枚举K=2^p，现实现Möbius/zeta为dense K×K矩阵，单矩阵内存4×4^p字节；20个伙伴仅这两矩阵就需要8 TiB，未实际分配。','本审计回答枚举/编码资源成本，并未训练更多agent的MTE控制策略；低阶或采样近似仍需独立实现和误差/控制验证，不替换原方法。']
    (out/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=P/'results'/p['name'];dest.mkdir(exist_ok=False)
    for name in ['RESULTS_CN.md','summary.json','protocol.json','source_manifest.json']:(dest/name).write_bytes((out/name).read_bytes())
    atomic_json(out/'status.json',dict(status='complete',source_checks=True,resource_cells=len(rows),environment_interfaces=len(env_checks)))
    print(out)
if __name__=='__main__':main()
