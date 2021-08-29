import os,sys
import numpy as np
import cPickle as pickle
import histogram as hist
import shutil
import easyplots as esp

# inputs
inpath = '/home/data/output/20201205/20201205_SK_BA30N5S5_Gplot/'
fn = 'SK_G_BA30N5S5_20201205.pkl'
cmap = 'jet'

infile = inpath + fn
d = pickle.load(open(infile,'r'))

# output
outpath = inpath
if not os.path.isdir(outpath):
        os.makedirs(outpath)
shutil.copy2(os.path.realpath(__file__), outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))


rnti = np.full((24, 33), float('nan'))
gc   = np.full((24, 33), float('nan'))
beta = np.full((24, 33), float('nan'))
tc   = np.full((24, 33), float('nan'))
psat = np.full((24, 33), float('nan'))
t=270.

for col in range(24):
	for row in range(33):
		try:
			rnti[col, row] = d[2]['r%dc%d'%(row, col)]['rnti']
			gc[col, row] = d[2]['r%dc%d'%(row, col)]['Gc']
			tc[col, row] = d[2]['r%dc%d'%(row, col)]['Tc']
			beta[col, row] = d[2]['r%dc%d'%(row, col)]['beta']
			psat[col, row] = gc[col, row]/(beta[col, row]+1.)*(tc[col, row]**(beta[col, row]+1)-t**(beta[col, row]+1))/tc[col, row]**beta[col, row]*1e-3
		except:
			continue



#im = 4
for im in [0]:
	hist.plot_1Dhist((rnti[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_rnti_hist'%im,
			maintitle=r'$R_n(Ti)$, module %d'%im,
			xlabel=r'$R_n(Ti)$', 
			xunit=r'm$\Omega$',
			binrange=[0., 250.])
	
	hist.plot_1Dhist((psat[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_psat_hist'%im,
	                maintitle=r'$P_{sat,270K}$, module %d'%im,
	                xlabel=r'$P_{sat,270K}$',
	                xunit=r'pW',
	                nbins=20,
	                binrange=[0., 5.])
	
	hist.plot_1Dhist((gc[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_gc_hist'%im,
	                maintitle=r'$G_{c}$, module %d'%im,
	                xlabel=r'$G_{c}$',
	                xunit=r'pW/K',
	                nbins=20,
	                binrange=[0., 30.])

	hist.plot_1Dhist((tc[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_tc_hist'%im,
	                maintitle=r'$T_{c}$, module %d'%im,
	                xlabel=r'$T_{c}$',
	                xunit=r'mK',
	                nbins=20,
	                binrange=[490., 530.])

	hist.plot_1Dhist((beta[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_beta_hist'%im,
	                maintitle=r'$beta$, module %d'%im,
	                xlabel=r'$beta$',
	                xunit=r' ',
	                nbins=20,
	                binrange=[1.6, 2.])




