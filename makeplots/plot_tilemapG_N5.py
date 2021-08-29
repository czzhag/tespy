import os,sys
import numpy as np
import cPickle as pickle
import triangle_mapping_30 as tri
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
module='N5'
for im in [0]:

	tri.triange_mapping_single(rnti[im*2:im*2+2], itile = 0,
                        vmin = 0,
                        vmax = 250.,
                        vari = r'$R_n$',
                        unit = r'm$\Omega$',
                        outpath = outpath + '%s_rnti.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'$R_n(Ti)$, %s'%module,
                        cmap = cmap)

	tri.triange_mapping_single(psat[im*2:im*2+2], itile = 0,
                        vmin = 0,
                        vmax = 5.,
                        vari = r'$P_{sat,270K}$',
                        unit = r'pW',
                        outpath = outpath + '%s_psat.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'$P_{sat,270K}$, %s'%module,
                        cmap = cmap)

        tri.triange_mapping_single(gc[im*2:im*2+2], itile = 0,
                        vmin = 0,
                        vmax = 30.,
                        vari = r'$G_{c}$',
                        unit = r'pW/K',
                        outpath = outpath + '%s_gc.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'$G_{c}$, %s'%module,
                        cmap = cmap)	

        tri.triange_mapping_single(tc[im*2:im*2+2], itile = 0,
                        vmin = 490,
                        vmax = 530.,
                        vari = r'$T_{c}$',
                        unit = r'mK',
                        outpath = outpath + '%s_tc.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'$T_{c}$, %s'%module,
                        cmap = cmap) 

        tri.triange_mapping_single(beta[im*2:im*2+2], itile = 0,
                        vmin = 1.6,
                        vmax = 2.0,
                        vari = r'$beta$',
                        unit = r' ',
                        outpath = outpath + '%s_beta.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'$beta$, %s'%module,
                        cmap = cmap) 






