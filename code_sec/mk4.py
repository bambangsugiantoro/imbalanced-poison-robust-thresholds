s = open('cq3_run.py').read()
old = "                        thrs['ideal'] = float(np.quantile(negt, q)) if len(negt) else thrs['naive']"
assert s.count(old) == 1
new = """                        aud_n = {}
                        for fr in AUD_FRACS:
                            rs2 = np.random.RandomState(s_['seed'] + 177); asel = rs2.rand(len(iva)) < fr; aneg = pv[asel & (y[iva] == 0)]
                            thrs['aud_%g' % fr] = float(np.quantile(aneg, q)) if len(aneg) >= 5 else thrs['naive']
                            thrs['audnp_%g' % fr] = npu_thr(aneg, alpha) if len(aneg) >= 5 else thrs['naive']
                            aud_n['aud_%g' % fr] = aud_n['audnp_%g' % fr] = int(len(aneg))
""" + old
s = s.replace(old, new)
s = s.replace("TRUST_FRAC = float(os.environ.get('TRUST_FRAC', '0.05'))", "TRUST_FRAC = float(os.environ.get('TRUST_FRAC', '0.05')); AUD_FRACS = [float(x) for x in os.environ.get('AUD_FRACS', '0.002,0.005,0.01,0.02,0.05').split(',')]")
s = s.replace("status=st, n_pos_test", "status=st, n_aud=aud_n.get(var, -1), n_pos_test")
open('cq4_run.py', 'w').write(s)
print('ok')
