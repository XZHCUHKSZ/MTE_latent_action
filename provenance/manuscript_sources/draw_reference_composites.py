"""Reference-inspired six-panel composite and four-panel resource/performance figure.
Only archived outcomes are plotted. No new training, smoothing, or seed filtering.
"""
from pathlib import Path
import json, os
import numpy as np
from scipy.stats import t
P=Path(__file__).resolve().parents[1];E=P/'evidence';F=P/'pictures'
os.environ['MPLCONFIGDIR']=str(P/'qa/mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':8.3,
 'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'axes.edgecolor':'#6C7279','text.color':'#111111','axes.labelcolor':'#111111',
 'pdf.fonttype':42,'savefig.pad_inches':.03})
def read(n):return json.loads((E/n).read_text(encoding='utf8'))
data=read('figure_numeric_current.json');scale=read('figure_scaling_current.json')
idx={(r['environment'],r['n'],r['budget'],r['method']):r for r in data}
COL={'anchor_edge_h1':'#C37436','anchor_graph':'#865BA5','anchor_mif':'#157E79',
 'anchor_tree':'#B14F68','lapo_joint':'#3D7EA6','laom_joint_k3':'#AD942C','bc_scarce':'#414A52'}
NAM={'anchor_edge_h1':'Edge-h1-Aux','anchor_graph':'Graph-Aux','anchor_mif':'MIF-Aux',
 'anchor_tree':'Tree-Aux','lapo_joint':'LAPO-joint','laom_joint_k3':'LAOM-joint','bc_scarce':'BC'}
COL['anchor_simple']='#557D65'
NAM['anchor_simple']='Simple-Aux'
NAM['anchor_edge_h1']='Edge-Aux (h2/h1)'
audit=[]
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),bbox_inches='tight',dpi=250)
 plt.close(fig)
def record(panel,series,values,**kw):
 audit.append(dict(panel=panel,series=series,seed_values=np.asarray(values).tolist(),**kw))

# Figure 2: explicit method rows, plus the separate full-system learning-route comparison.
fig=plt.figure(figsize=(6.25,4.65))
fig.text(.005,.985,'(a) Control performance and auxiliary-route ablation',fontsize=9)
family=['anchor_edge_h1','anchor_edge_h3','anchor_simple','anchor_graph','anchor_tree','anchor_mif','lapo_joint','laom_joint_k3']
family_names=['Edge-h1-Aux','Edge-h3-Aux','Simple-Aux','Graph-Aux','Tree-Aux','MIF-Aux','LAPO-joint','LAOM-joint']
for env,n,rect,title in [
 ('mpe',700,[.155,.732,.292,.214],'(i) MPE: formal control'),
 ('mamujoco',90,[.669,.732,.285,.214],'(ii) MaMuJoCo: formal control')]:
 ax=fig.add_axes(rect);ys=[0,1,2,3,4,5,6.4,7.4];means=[]
 for i,m in enumerate(family):
  bs=sorted(r['budget'] for r in data if r['environment']==env and r['n']==n and r['method']==m)
  y=np.mean([idx[(env,n,b,m)]['seed_returns'] for b in bs],axis=0)
  mu=float(y.mean());sd=float(y.std(ddof=1));means.append(mu)
  co=COL.get(m,'#8C789E')
  ax.errorbar(mu,ys[i],xerr=sd,fmt='s' if m.startswith(('lapo','laom')) else 'o',color=co,ms=3,capsize=2,lw=1.0)
  record('Fig2 '+env,m,y,method=m,mean=mu,seed_sd=sd,budgets=bs)
 best=int(np.argmax(means))
 for i,mu in enumerate(means):
  ax.text(1.025,ys[i],f'{mu:.3f}' if env=='mpe' else f'{mu:.1f}',transform=ax.get_yaxis_transform(),va='center',fontsize=6.7,fontweight='bold' if i==best else 'normal')
 ax.axhline(5.7,color='#CCCCCC',lw=.5,ls=':')
 ax.set_yticks(ys,[s.replace('Edge-h1','Edge-h2') if env=='mpe' else s for s in family_names],fontsize=7.0);ax.set_ylim(7.95,-.5)
 ax.get_yticklabels()[best].set_fontweight('bold')
 ax.set_xlabel('Budget-averaged return ↑',fontsize=7.2,labelpad=2)
 ax.set_title(title,loc='left',pad=3,fontsize=8.1);ax.grid(axis='x',alpha=.14)
 ax.tick_params(axis='y',length=0,pad=2)
 ax.set_xlim((-17.62,-17.10) if env=='mpe' else (280,785))
 ax.set_xticks([-17.6,-17.4,-17.2] if env=='mpe' else [300,450,600,750])
fig.text(.005,.638,'Graph / Simple / Tree-Aux: mean gains at 6/6 budgets.',fontsize=6.9)
fig.text(.515,.638,'MIF-Aux: above both latent baselines at all 5 budgets.',fontsize=6.9)

# (iii) MPE return gains: show both upstream seeds, never count decoder repeats as seeds.
abl=read('control_ablation_current.json')
ax=fig.add_axes([.138,.423,.32,.147]);ax.axvline(0,color='#777777',lw=.65,ls='--')
for i,r in enumerate(abl['mpe']):
 y=np.array(r['upstream_seed_mean_gains']);co={'mif':'#157E79','graph':'#865BA5','simple':'#557D65','tree':'#B14F68'}[r['method']]
 ax.plot([0,y.mean()],[i,i],color=co,lw=.75,alpha=.6,zorder=2)
 ax.scatter(y,np.full(y.shape,i),facecolors='white',edgecolors=co,linewidths=.7,s=10,zorder=3,marker='o')
 ax.scatter([y.mean()],[i],color=co,linewidths=.4,s=16,zorder=4,marker='D')
 ax.text(1.015,i,f'{y.mean():+.3f}',transform=ax.get_yaxis_transform(),va='center',fontsize=7)
 record('Fig2 iii',r['method'],y,decoder_repeats=1,upstream_seeds=[45,46,47,48,49],comparator='Base + Global-Joint16')
ax.set_yticks([0,1,2,3],['MIF-Aux','Graph-Aux','Simple-Aux','Tree-Aux']);ax.set_ylim(3.5,-.5);ax.set_xlim(-.23,.28)
ax.set_xticks([-.2,0,.2]);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0)
ax.set_xlabel('Return gain over Base + Global-Joint16 ↑',labelpad=2,fontsize=6.8)
fig.text(.005,.607,'(iii) MPE: auxiliary-route control gains',fontsize=8.0)
fig.text(.005,.343,'Five training seeds; all six budgets within each seed.',fontsize=6.7)

# (iv) MaMuJoCo keeps the complete budget grid, including the negative B16 mean.
ax=fig.add_axes([.635,.423,.344,.147]);r=abl['mamujoco'];bs=r['budgets'];y=np.array(r['upstream_seed_mean_gains_by_budget']);x=np.arange(5)
ax.axhline(0,color='#777777',lw=.65,ls='--')
for i,(co,mark) in enumerate([('#A6C6BF','o'),('#C1B5D2','s'),('#CDB997','^'),('#97B4CD','v'),('#CFA6A6','x')]):
 ax.plot(x,y[i],ls=':',marker=mark,color=co,ms=2.5,lw=.8,label='Seed '+str(i))
ax.plot(x,y.mean(0),'-D',color='#157E79',ms=3,lw=1.3,label='Mean')
ax.set_xticks(x,bs);ax.set_ylim(-45,35);ax.set_yticks([-40,-20,0,20]);ax.grid(axis='y',alpha=.15)
ax.set_ylabel('Return gain ↑',labelpad=1,fontsize=7);ax.set_xlabel('Labelled episodes B',labelpad=2,fontsize=7)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=6,frameon=False,fontsize=5.7,columnspacing=.4,handlelength=.8)
fig.text(.555,.607,'(iv) MaMuJoCo: MIF-Aux (mean gain +1.63)',fontsize=8.0)
fig.text(.555,.343,'Five upstream seeds; five decoder repeats per seed.',fontsize=6.7)
record('Fig2 iv','MIF-Aux versus Base + Global-Joint16',y,action_budgets=bs,decoder_repeats=5,upstream_seeds=[0,1,2,3,4])

fig.text(.005,.307,'(b) Visual task: recorded shared-body motion and control',fontsize=9,weight='normal')
rgb=np.load(E/'illustration_rgb.npy');assert rgb.shape==(5,4,64,64,3)
gs=fig.add_gridspec(2,4,left=.005,right=.417,bottom=.041,top=.238,wspace=.06,hspace=.37)
for j in range(4):
 ax=fig.add_subplot(gs[0,j]);ax.imshow(rgb[2,j],interpolation='nearest');ax.axis('off');ax.set_title('View '+str(j),fontsize=7.2,pad=2)
 ax=fig.add_subplot(gs[1,j]);ax.imshow(rgb[[0,1,3,4][j],0],interpolation='nearest');ax.axis('off');ax.set_title('$t=%d$'%[0,40,80,120][j],fontsize=7.2,pad=2)
fig.text(.005,.277,'(i) Recorded views and motion',fontsize=7.9,weight='normal')
fig.text(.005,.010,'Synchronized views at $t=60$; fixed training timestamps.',fontsize=6.5)
ax=fig.add_axes([.58,.09,.395,.12]);v=read('visual_low_label_summary.json')['records']
arms=[('anchor_solo_edge_cara_mif','MIF-Solo','#157E79'),('anchor_plus_edge_cara_mif','MIF-Aux','#557D65'),('anchor_only','Base','#865BA5'),('bc_batch256','BC','#414A52'),('bc_idm_relabel','IDM','#C37436')]
for arm,label,co in arms:
 y=np.array([[r['mean_return'] for r in v if r['arm']==arm and r['budget']==b] for b in [1,2,4,8]])
 assert y.shape==(4,5)
 mu=y.mean(1);sd=y.std(1,ddof=1);x=np.arange(4)
 ax.plot(x,mu,'-o',color=co,ms=2,lw=1,label=label)
 ax.fill_between(x,mu-sd,mu+sd,color=co,alpha=.08,lw=0)
 record('Fig2 b ii low-label',arm,y,budgets=[1,2,4,8],labels=[200,400,800,1600])
ax.set_xticks(range(4),['200','400','800','1600']);ax.set_yticks([0,80,160]);ax.grid(axis='y',alpha=.14)
ax.set_xlabel('Target-action labels',labelpad=1,fontsize=7);ax.set_ylabel('Return ↑',labelpad=1,fontsize=7)
ax.legend(loc='lower left',bbox_to_anchor=(-.04,1.01),borderaxespad=0,ncol=3,frameon=False,fontsize=5.8,columnspacing=.65,handlelength=1)
fig.text(.555,.277,'(ii) Visual label efficiency: mean ± seed SD',fontsize=7.7)
fig.text(.555,.010,'Same five frozen RGB frontends; shared evaluation conditions.',fontsize=6.1)
save(fig,'fig2_results_composite')

if os.environ.get('MTE_FIG2_ONLY') == '1':
 (E/'figure2_ablation_update_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 raise SystemExit(0)

# Figure 3: compact 2x2 curves. Remove redundant twin axes and final-point stars.
# Keep every budget and the five-seed SD for every displayed series.
fig,axs=plt.subplots(2,2,figsize=(6.25,3.05))
fig.subplots_adjust(left=.083,right=.985,bottom=.24,top=.964,wspace=.23,hspace=.58)
panel_specs=[('mpe',700,'anchor_graph','control'),('mamujoco',90,'anchor_mif','control'),
             ('mpe',700,'anchor_graph','scaling'),('mamujoco',90,'anchor_edge_h1','scaling')]
for k,(ax,(env,n,focal,mode)) in enumerate(zip(axs.flat,panel_specs)):
 methods=['anchor_edge_h1','anchor_simple','anchor_graph','anchor_tree','anchor_mif','lapo_joint','laom_joint_k3','bc_scarce']
 for method in methods:
  if mode=='scaling' and not any(r['environment']==env and r['method']==method for r in scale): continue
  if mode=='control':
   rr=sorted([r for r in data if r['environment']==env and r['n']==n and r['method']==method],key=lambda r:r['budget'])
   xs=[r['budget'] for r in rr];y=np.array([r['seed_returns'] for r in rr]).T
  else:
   r=next(r for r in scale if r['environment']==env and r['method']==method);xs=r['n_values'];y=np.array(r['seed_returns_by_n'])
   assert np.allclose(y[:,-1],idx[(env,n,r['budget'],method)]['seed_returns'])
  assert y.shape==(5,len(xs));x=np.arange(len(xs));mu=y.mean(0);sd=y.std(0,ddof=1);co=COL[method]
  ax.fill_between(x,mu-sd,mu+sd,color=co,alpha=.065,lw=0)
  ax.plot(x,mu,linestyle='--' if method=='bc_scarce' else '-',marker='s' if method=='anchor_mif' else 'o',color=co,ms=2.2,lw=1.35 if method==focal else .95)
  record('Fig3 '+str(k),method,y,x=xs,x_is='action_label_budget' if mode=='control' else 'unlabelled_episodes')
 ax.set_xticks(x,xs);ax.set_xlabel('Labelled episodes B' if mode=='control' else 'Unlabelled episodes N',labelpad=.5,fontsize=7)
 ax.set_ylabel('Return (higher is better)',labelpad=2,fontsize=7);ax.grid(axis='y',alpha=.15,lw=.5)
 titles=['(a) MPE: label budget','(b) MaMuJoCo: label budget','(c) MPE: fixed B = 32','(d) MaMuJoCo: fixed B = 8']
 ax.set_title(titles[k],pad=2,fontsize=8.2)
 ax.tick_params(axis='both',labelsize=7,length=2,pad=2)
legend_methods=['anchor_edge_h1','anchor_simple','anchor_graph','anchor_tree','anchor_mif','lapo_joint','laom_joint_k3','bc_scarce']
handles=[Line2D([0],[0],color=COL[m],ls='--' if m=='bc_scarce' else '-',lw=1.1,marker='o',ms=2.3) for m in legend_methods]
fig.legend(handles,[NAM[m] for m in legend_methods],loc='lower center',bbox_to_anchor=(.51,.037),ncol=4,frameon=False,fontsize=7.2,columnspacing=1.0,handlelength=1.4,labelspacing=.3)
fig.text(.5,.005,'Fixed labels: Simple / Graph / Tree-Aux gain 0.179–0.247 in MPE; Edge / Graph / Tree gain 102–112 in Ant.',ha='center',fontsize=7.0)
save(fig,'fig3_resource_performance')

(E/'reference_composite_data.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print('Wrote two composite PDFs from',len(audit),'audited data series.')
