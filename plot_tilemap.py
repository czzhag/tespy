import os,sys
import numpy as np
import cPickle as pickle
import triangle_mapping_30 as tri
import shutil
# inputs
run = '20200102_BA_BA30N1N2N3N4'
module = 'N4'
im = 10
inpath = './output/20200102/%s/'%run
fn = '%s_dpdt_rnti.pkl'%run
cmap = 'jet'

infile = inpath + fn
d = pickle.load(open(infile,'r'))
# output
outpath = './output/20200102/tilemaps/'
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
                        vmin = 0.02,
                        vmax = 0.07,
                        vari = r'dP/dT',
                        unit = r'pW/K',
                        outpath = outpath + '%s_dpdt.png'%module,
                        mask_wire = None,
                        mapping='BA',
                        maintitle= r'dP/dT, %s'%module,
                        cmap = cmap)

tri.triange_mapping_single(rnti[im*2:im*2+2], itile = 0, 
			vmin = 0,
			vmax = 250.,
			vari = r'$R_n$',
			unit = r'm$\Omega$',
			outpath = outpath + '%s_rnti.png'%module, 
			mask_wire = None, 
			mapping='BA', 
			maintitle= r'$R_n(Ti)$, %s'%module, 
			cmap = cmap)
