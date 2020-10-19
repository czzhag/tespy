import os,sys
import numpy as np
import cPickle as pickle
import histogram as hist
import shutil
import easyplots as esp

# inputs
inpath = '/home/czhang/analysispy/tespy/output/20200122/20200122_BA_BA1_plot/'
fn = '20200122_BA_BA1.pkl'
cmap = 'jet'

infile = inpath + fn
d = pickle.load(open(infile,'r'))
# output
outpath = inpath.replace('cryo', 'output')
if not os.path.isdir(outpath):
        os.makedirs(outpath)
shutil.copy2(os.path.realpath(__file__), outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))

rnti = np.full((24, 33), float('nan'))
psat = np.full((24, 33), float('nan'))

for col in range(24):
	for row in range(33):
		try:
			rnti[col, row] = d[1]['r%dc%d'%(row, col)][2]*1e3
			psat[col, row] = d[1]['r%dc%d'%(row, col)][3]*1e12
		except:
			continue



#im = 4
for im in range(12):
    hist.plot_1Dhist((rnti[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_rnti_hist'%im,
    		maintitle=r'$R_n(Ti)$, module %d'%im,
    		xlabel=r'$R_n(Ti)$', 
    		xunit=r'm$\Omega$',
    		binrange=[50., 180.])
    
    fig,ax = esp.presetting(7.4,6,lx="channels",ly="Rn, mOhm")
    ax.scatter(range(66), (rnti[im*2:im*2+2, :]).reshape(2*33))
    esp.possetting(fig, ffn = outpath+'module%d_rnti_scatter.png'%im, ifleg = False, ifgrid = True, ifshow = False)

    hist.plot_1Dhist((psat[im*2:im*2+2, :]).reshape(2*33), outpath, 'module%d_psat_hist'%im,
                    maintitle=r'$P_{sat}$, module %d'%im,
                    xlabel=r'$P_{sat}$',
                    xunit=r'pW',
                    nbins=20,
                    binrange=[0., 5.])

    fig,ax = esp.presetting(7.4,6,lx="channels",ly="Psat, pW")
    ax.scatter(range(66), (psat[im*2:im*2+2, :]).reshape(2*33))
    esp.possetting(fig, ffn = outpath+'module%d_psat_scatter.png'%im, ifleg = False, ifgrid = True, ifshow = False)
