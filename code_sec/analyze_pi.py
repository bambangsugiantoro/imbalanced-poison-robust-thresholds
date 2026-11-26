import sys, glob, os, numpy as np, pandas as pd
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D = D[(D['var'] == 'naive') & (D.poison > 0) & (D.op == D.op.iloc[0])]
D['e_cq'] = (D.pi_hat - D.pi_true).abs(); D['e_tilt'] = (D.pi_tilt - D.pi_true).abs()
print(D.groupby(['dataset', 'noise'])[['pi_true', 'pi_hat', 'pi_tilt', 'e_cq', 'e_tilt']].median().round(4).to_string())
print('share tilt closer:'); print(D.assign(c=D.e_tilt < D.e_cq).groupby(['dataset', 'noise'])['c'].mean().round(3).to_string())
