# find common bias for each MCE col

import os,sys
import numpy as np
import mce_data
import lcdata as lc
import calib_BA1
import calib_SK
sys.path.insert(0, "/home/cheng/analysis/ploting")
import matplotlib.pyplot as plt
import easyplots as esp
import shutil

ncol = 24
nrow = 33

def main():
	inpath = '/home/data/cryo/20191023/LC_G_FPU_271mK_datamode1_run1'
	modulename = 'M6'
	calfn = 'calib_SK'
	fitrange = {}
	fitrange['rnti_low'] = 4000
	fitrange['rnti_hgh'] = 8000
	fitrange['sc_low']   = 0
	fitrange['sc_hgh']   = 500
	cols = [0,1]
	plt.close('all')
	findbias_flatpr(inpath, modulename, calfn, fitrange, cols, biasmin = 500, biasmax=4000, flatrange=0.01)

def findbias_flatpr(inpath, modulename, calfn = 'calib_BA1', fitrange=None, cols=None, rows=None, 
			biasmin=None, biasmax=None, rrnpsat=0.6, flatrange=0.01):
	lcfn = inpath.split('/')[-1]
	fbfn = inpath + '/' + lcfn
	biasfn = fbfn + '.bias'
	f = mce_data.MCEFile(fbfn)
	y = -1.0*f.Read(row_col=True,unfilter='DC').data
	bias = np.loadtxt(biasfn,skiprows=1)
	
	outpath = inpath.replace('cryo', 'output') + '/findbias_flatpr/'
	shutil.copy2(os.path.realpath(__file__),
        	outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
	if not os.path.isdir(outpath):
		os.makedirs(outpath)

	biascounts = np.zeros((ncol, nrow, len(bias)))
	roptmatrix = np.full((ncol, nrow, len(bias)), float('nan'))
	rntis      = np.full((ncol, nrow), float('nan'))
	if not fitrange:
		print('** NO FITRANGE USE DEFAULT **')
		fitrange = ls.get_default_fitrange()
	if calfn == 'calib_SK':
		calib = calib_SK.calib_sk()
	elif calfn == 'calib_BA1':
		calib = calib_BA1.calib_ba1()
	else:
		print('** WRONG CALIB FILENAME **')
		print('** USE DEFAULT BA1 **')
		calib = calib_BA1.calib_ba1()

	if not cols:
		cols = range(ncol)
	if not rows:
		rows = range(nrow)

	for col in cols:
		for row in rows:
			biascalib, fbcalib, rnti, ksc = lc.get_LC(bias, y[row,col], calib, fitrange)
			rr, pp, psat = lc.get_PR_Ti(biascalib, fbcalib, calib, rnti, rnti*rrnpsat)
			#print(rr*1000)
			#print(ksc)
			#print(rnti*rrnpsat)
			#print(np.where((pp<psat*(1+flatrange))&(pp>psat*(1-flatrange))))
			if np.isnan(rnti) or np.isnan(psat):
				continue
			biascounts[col, row, np.where((pp<psat*(1+flatrange))&(pp>psat*(1-flatrange)))[0]] = 1
			roptmatrix[col, row] = rr
			rntis[col, row]      = rnti
			
			fig,ax = esp.presetting(6, 5, 'r', 'p')
			ax.plot(rr*1000, pp*1e12)
			ax.axhline(y=psat*(1+flatrange)*1e12, color='r')
			#ax.axhline(y=psat*(1)*1e12, color='g')
			ax.scatter([rnti*rrnpsat*1000], [psat*(1)*1e12], s=80, marker='+', color='g')
			ax.axhline(y=psat*(1-flatrange)*1e12, color='r')
			try:
				ax.set_xlim(0, rnti*1.2*1000)
				ax.set_ylim(psat*1e12-0.5, psat*1e12+0.5)
			except:
				continue
			esp.possetting(fig, ffn = outpath+'c%dr%d.png'%(col,row), ifshow=False)
		
	biascounts_pcol = np.sum(biascounts, axis=1)

	plot_findbias_R(bias, biascounts, biascounts_pcol, outpath, modulename, roptmatrix, rntis, cols, rows, biasmin, biasmax)

	return bias, biascounts, biascounts_pcol, roptmatrix, rntis

def plot_findbias(bias, biascounts, biascounts_pcol, outpath, modulename, roptmatrix, rntis=None, cols=None, rows=None,
			biasmin=None, biasmax=None):
	if not cols:
		cols = range(ncol)
	if not rows:
		rows = range(nrow)
	# plot bias counts
	biascountsnan = np.where(biascounts==0, float('nan'), biascounts)
	for col in cols:
		fig, ax = esp.premultps(1, 3, 4, 6, ['bias, ADU', 'bias, ADU', r'$R_{opt}, m\Omega$'], ['row', 'nrows', ' '])
		for row in rows:
			ax[0].plot(bias[::-1], (biascountsnan[col, row, ::-1]*row), color = 'k')
			ax[0].set_ylim(min(rows)-0.5, max(rows)+0.5)
		ax[1].plot(bias, biascounts_pcol[col, :], color = 'k')
		if biasmax and biasmin:
			ax[0].set_xlim(biasmin, biasmax)
			ax[1].set_xlim(biasmin, biasmax)
		biasp = np.nanmean(bias[np.where(biascounts_pcol[col, :]>np.nanmax(biascounts_pcol[col, :])*0.8)])
		ax[0].axvline(x=biasp, color='r', label = 'selected bias')
		ax[1].axvline(x=biasp, color='r', label = 'selected bias')
		Ropts = np.full(nrow, float('nan'))
		for row in rows:
			Ropts[row] = np.interp(biasp, bias[::-1], roptmatrix[col,row,::-1])
		ax[2].hist(Ropts*1000, range=[0,100], histtype='stepfilled', color='r', label=r'$R_{opt}$ at the selected bias')
		esp.posmultps(fig, ax, outpath + 'findbias_barhist_%s_col%d.png'%(modulename,col), ifshow=False, ifleg=True)

		


def plot_findbias_R(bias, biascounts, biascounts_pcol, outpath, modulename, roptmatrix, rntis=None, cols=None, rows=None,
			biasmin=None, biasmax=None):
	if not cols:
		cols = range(ncol)
	if not rows:
		rows = range(nrow)
	# plot bias counts
	biascountsnan = np.where(biascounts==0, float('nan'), biascounts)
	for col in cols:
		fig, ax = esp.premultps(1, 3, 4, 6, ['bias, ADU', 'bias, ADU', r'$R_{opt}, m\Omega$'], ['row', 'nrows', 'row'])
		for row in rows:
			ax[0].plot(bias[::-1], (biascountsnan[col, row, ::-1]*row), color = 'k')
			ax[0].set_ylim(min(rows)-0.5, max(rows)+0.5)
			ax[2].plot(roptmatrix[col,row,::-1]*1000, (biascountsnan[col, row, ::-1]*row), color = 'k')
                        ax[2].set_ylim(min(rows)-0.5, max(rows)+0.5)
		ax[1].plot(bias, biascounts_pcol[col, :], color = 'k')
		if biasmax and biasmin:
			ax[0].set_xlim(biasmin, biasmax)
			ax[1].set_xlim(biasmin, biasmax)
		biasp = np.nanmean(bias[np.where(biascounts_pcol[col, :]>np.nanmax(biascounts_pcol[col, :])*0.8)])
		for row in rows:
			ax[2].scatter(np.interp(biasp, bias[::-1], roptmatrix[col,row,::-1])*1000, row, color='r')
		ax[0].axvline(x=biasp, color='r', label = 'selected bias')
		ax[1].axvline(x=biasp, color='r', label = 'selected bias')		

		#Ropts = np.full(nrow, float('nan'))
		#for row in rows:
		#	Ropts[row] = np.interp(biasp, bias[::-1], roptmatrix[col,row,::-1])
		#ax[2].hist(Ropts*1000, range=[0,100], histtype='stepfilled', color='r', label=r'$R_{opt}$ at the selected bias')
		
		esp.posmultps(fig, ax, outpath + 'findbias_barropt_%s_col%d.png'%(modulename,col), ifshow=False, ifleg=True)

		


if __name__ == '__main__':
	main()	
