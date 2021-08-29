import os,sys
import numpy as np
import cPickle as pickle
import triangle_mapping as tri
import shutil

# inputs
module = 'Mx1'
run = '20201016_SK_BA40Mx1T3_mesa'
im = 4 
inpath = '/home/data/output/20201016/%s/'%run
fn = '%s_dpdt_rnti.pkl'%run
cmap = 'jet'
infile = inpath + fn
d = pickle.load(open(infile,'r'))

# output
outpath = '/home/data/output/20201016/%s/tilemaps/'%run
if not os.path.isdir(outpath):
        os.makedirs(outpath)
shutil.copy2(os.path.realpath(__file__), outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))

#rnti = np.full((24, 33), float('nan'))
#dpdt = np.full((24, 33), float('nan'))
rnti = np.array([[np.nan for i2 in range(33)] for i1 in range(24)])
dpdt = np.array([[np.nan for i2 in range(33)] for i1 in range(24)])


for col in range(24):
	for row in range(33):
		try:
			rnti[col, row] = d[1][col][row]
			dpdt[col, row] = d[0][col][row]
		except:
			continue


tri.triange_mapping_single(dpdt[im*2:im*2+2], itile = 0,
                        vmin = 0.03,
                        vmax = 0.08,
                        vari = r'dP/dT',
                        unit = r'pW/K',
                        outpath = outpath + '%s_dpdt.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'dP/dT, %s'%module,
                        cmap = cmap)

tri.triange_mapping_single(rnti[im*2:im*2+2], itile = 0, 
			vmin = 50,
			vmax = 250.,
			vari = r'$R_n$',
			unit = r'm$\Omega$',
			outpath = outpath + '%s_rnti.png'%module, 
			mask_wire = None, 
			mapping='SK', 
			maintitle= r'$R_n(Ti)$, %s'%module, 
			cmap = cmap)
