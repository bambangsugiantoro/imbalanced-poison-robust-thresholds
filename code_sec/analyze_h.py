"""H2/H4 diagnostics. usage: python analyze_h.py DIR [DIR2...]"""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
fs=[f for d in sys.argv[1:] for f in glob.glob(os.path.join(d,'*.csv')) if '.partial' not in f]
D=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
D['cal']=np.log2(np.maximum(D['FPR'],1e-4)/D['target'])
print('== H4: pi_hat error by validation AUC (CQ rows, poison>0) ==')
c=D[(D['var']=='cq')&(D['poison']>0)].copy(); c['err']=(c['pi_hat']-c['pi_true']).abs()
c['bin']=pd.cut(c['val_auc'],[0,0.7,0.9,0.97,1.01],labels=['<0.7','0.7-0.9','0.9-0.97','>=0.97'])
print(c.groupby('bin').agg(n=('err','size'),pi_hat=('pi_hat','mean'),pi_true=('pi_true','mean'),mae=('err','mean'),share_within_0p1=('err',lambda x:(x<=0.1).mean())).round(3).to_string())
print('\n== CQ vs naive abs calibration by validation AUC bin (poison>0) ==')
k=['dataset','split','poison','learner','strategy','op']
w=D[D['poison']>0].pivot_table(index=k,columns='var',values=['cal','val_auc'],aggfunc='first')
x=pd.DataFrame({'auc':w[('val_auc','naive')],'a_cq':w[('cal','cq')].abs(),'a_nv':w[('cal','naive')].abs(),'a_or':w[('cal','oracle')].abs()}).dropna()
x['bin']=pd.cut(x['auc'],[0,0.7,0.9,0.97,1.01],labels=['<0.7','0.7-0.9','0.9-0.97','>=0.97'])
print(x.groupby('bin').agg(n=('a_cq','size'),cq=('a_cq','mean'),naive=('a_nv','mean'),oracle=('a_or','mean'),cq_better=('a_cq',lambda s:(s<x.loc[s.index,'a_nv']).mean())).round(3).to_string())
print('\n== H2: Spearman(recall, FPR) across threshold rules on the same model ==')
r=[]
for key,g in D[D['var'].isin(['naive','oracle','cq','cap25','gcap25','ideal'])].groupby(k):
    if g['FPR'].nunique()>2 and g['Recall'].nunique()>2: r.append(spearmanr(g['FPR'],g['Recall'])[0])
r=np.array(r); print('groups',len(r),'mean rho %.3f median %.3f share rho>0.8 %.3f'%(np.nanmean(r),np.nanmedian(r),np.nanmean(r>0.8)))
