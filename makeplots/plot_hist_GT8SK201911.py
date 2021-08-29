import os,sys
import numpy as np
import cPickle as pickle
import histogram as hist

# inputs
inpath = '/home/data/output/20191109/20191109_SK_BA40T8_Gplot/'
fn = 'SK_G_BA40T8_20191109.pkl'
infile = inpath + fn
d = pickle.load(open(infile,'r'))

# output
outpath = inpath.replace('cryo', 'output')
if not os.path.isdir(outpath):
	os.makedirs(outpath)

gc = np.full((24, 33), float('nan'))
g450 = np.full((24, 33), float('nan'))
rnti = np.full((24, 33), float('nan'))
beta = np.full((24, 33), float('nan'))
tc = np.full((24, 33), float('nan'))
psat = np.full((24, 33), float('nan'))
t = 270.
for col in range(24):
	for row in range(33):
		try:
			rnti[col, row] = d[2]['r%dc%d'%(row, col)]['rnti']
			gc[col, row] = d[2]['r%dc%d'%(row, col)]['Gc']
			tc[col, row] = d[2]['r%dc%d'%(row, col)]['Tc']
			beta[col, row] = d[2]['r%dc%d'%(row, col)]['beta']
			g450[col, row] = gc[col, row]*(450./tc[col, row])**beta[col, row]
			psat[col, row] = gc[col, row]/(2.+1.)*(tc[col, row]**(2+1)-t**(2+1))/tc[col, row]**2*1e-3
		except:
			continue

im = 0
hist.plot_1Dhist((rnti[im*2:im*2+2, :]).reshape(2*33), outpath, 'M7_rnti_hist',
		maintitle=r'$R_n(Ti)$, M7 T8',
		xlabel=r'$R_n(Ti)$', 
		xunit=r'm$\Omega$',
		binrange=[50., 180.])

hist.plot_1Dhist((gc[im*2:im*2+2, :]).reshape(2*33), outpath, 'M7_gc_hist',
                maintitle=r'$G_c$, M7 T8',
                xlabel=r'$G_c$',
                xunit=r'pW/K',
                binrange=[10., 30.])

hist.plot_1Dhist((g450[im*2:im*2+2, :]).reshape(2*33), outpath, 'M7_g450_hist',
                maintitle=r'$G_{450}$, M7 T8',
                xlabel=r'$G_{450}$',
                xunit=r'pW/K',
                binrange=[10., 30.])

hist.plot_1Dhist((tc[im*2:im*2+2, :]).reshape(2*33), outpath, 'M7_tc_hist',
                maintitle=r'$T_c$, M7 T8',
                xlabel=r'$T_c$',
                xunit=r'mK',
                binrange=[480., 500.])

hist.plot_1Dhist((psat[im*2:im*2+2, :]).reshape(2*33), outpath, 'M7_psat270_hist',
                maintitle=r'$P_{sat}(270 mK)$, M7 T8',
                xlabel=r'$P_{sat}$',
                xunit=r'pW',
                binrange=[1., 4.])
