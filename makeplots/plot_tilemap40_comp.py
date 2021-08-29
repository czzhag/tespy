import os,sys
import numpy as np
import cPickle as pickle
import triangle_mapping as tri
import shutil

# inputs
module = 'Mx1'
run = '20201016_SK_BA40Mx1T3_mesa'
im = 4 
cmap = 'jet'
infile1 = '/home/data/output/20201016/20201016_SK_BA40Mx1T3_mesa/20201016_SK_BA40Mx1T3_mesa_dpdt_rnti.pkl'
infile2 = '/home/data/output/20181209/LC_light_ba_FPU315mK_datamode1_run_077K_2/BA40T3M1_dpdt_rnti.pkl' 
d1 = pickle.load(open(infile1,'r'))
d2 = pickle.load(open(infile2,'r'))

# output
outpath = '/home/data/output/20201016/%s/tilemaps_comp/'%run
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
			rnti[col, row] = d1[1][col][row]-d2[1][int(np.mod(col,2))][row]
			dpdt[col, row] = d1[0][col][row]-d2[0][int(np.mod(col,2))][row]
		except:
			continue


tri.triange_mapping_single(dpdt[im*2:im*2+2], itile = 0,
                        vmin = -0.03,
                        vmax = 0.03,
                        vari = r'dP/dT',
                        unit = r'pW/K',
                        outpath = outpath + '%s_dpdt.png'%module,
                        mask_wire = None,
                        mapping='SK',
                        maintitle= r'dP/dT now-then, %s'%module,
                        cmap = cmap)

tri.triange_mapping_single(rnti[im*2:im*2+2], itile = 0, 
			vmin = -50,
			vmax = 50.,
			vari = r'$R_n$',
			unit = r'm$\Omega$',
			outpath = outpath + '%s_rnti.png'%module, 
			mask_wire = None, 
			mapping='SK', 
			maintitle= r'$R_n(Ti)$ now-then, %s'%module, 
			cmap = cmap)
