import numpy as np
import sys, os
sys.path.insert(0, "/home/cheng/analysis/ploting")
sys.path.insert(0, "/home/cheng/analysis/timestream")
from tsdata import get_raw_timestream as get_fb
from tsdata import remove_baseline
import easyplots as esp
import calib_BA1
import calib_SK
import mce_data
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import signal
from scipy.fftpack import dct,dst,fft
import cPickle as pickle
import ba40_ModuleMapping as ba40map
import glob
import shutil
#from progress.bar import Bar
# Get noise power from time stream
# usage: python rowlenscan.py /home/data/cryo/20191001/ row_len_scan_rl*
# output in /home/data/output/20191001/rowlenscan_filt<constant>_nfft<10000>_at<0.1>

def main():
	#options
	#===============================
	Nfreq = 1
	detrend = 'constant' # for signal.welch
	nfft = 10000	     # for signal.welch
	targetpath = sys.argv[1]
	targetfiles = sys.argv[2]
	cols = range(24)         # the cols in interest. though the data structure will be saved in 24*33 matrix
	rows = range(33)
	#===============================	

	outpath = targetpath.replace('cryo', 'output') + '%s_filt%s_nfft%d_at%.1f/'%(targetfiles.replace('*','X'), detrend, nfft, Nfreq)
	if not os.path.isdir(outpath):
		os.makedirs(outpath)

	fnpickle = outpath+'rowlenscan_Np%.1f.pkl'%Nfreq
	if os.path.isfile(fnpickle):
		ifreproduce = raw_input('Reproduce?')
		shutil.copy2(os.path.realpath(__file__), outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
		if ifreproduce in [1, 'y', 'yes','Y','Yes']:
			reproduce(targetpath, targetfiles, Nfreq,cols,rows, detrend, nfft, outpath)
	else:
		
		reproduce(targetpath, targetfiles, Nfreq,cols,rows, detrend, nfft, outpath)

	outputs = pickle.load(open(fnpickle,'r'))
	plot_rowlenscan(outputs, cols, rows, Nfreq, outpath)

def reproduce(targetpath, targetfiles, Nfreq,cols,rows, detrend, nfft, outpath):
	filelist_ = glob.glob(targetpath+targetfiles)
	filelist = [fn for fn in filelist_ if not (fn.split('.')[-1] == 'run')]
	Np = np.full((len(filelist),24,33), float('nan')) # noise current at certain frequency (Nfreq)
	rowlens = np.full(len(filelist), float('nan')) 

	#bar = Bar("files:", max=len(filelist))
	for ii,datafn in enumerate(filelist):
		if datafn.split('.')[-1] == 'run':
			continue
		nn, ll = get_noise(datafn, cols, rows, Nfreq, detrend, nfft, ifplot=True, outpathps=outpath+'ps/')
		rowlens[ii] = ll
		Np[ii] = nn
		#bar.next()
	#bar.finish()
	pickle.dump((Np, rowlens),open(outpath+'rowlenscan_Np%.1f.pkl'%Nfreq,'w'))

# default: subtract mean, 'constant'
# detrend and nfft will be passed to signal.welch
# outputs: noise (24*33 array), rowlen float
def get_noise(datafn, cols, rows, Nfreq, detrend='constant', nfft=10000, ifplot=False, outpathps=None):
	fbs = get_fb(datafn, calfn='calib_BA1')
	rowlen = fbs.info['header']['row_len']
	fs = fbs.info['freq']
	noise = np.full((24,33), float('nan'))

	if ifplot and outpathps:
		if not os.path.isdir(outpathps):
			os.makedirs(outpathps)

	for mc in cols:
		for mr in rows:
			y = fbs.fbCalib[mr,mc]
			ff, ps = signal.welch(y, fs, return_onesided=True, detrend=detrend, nperseg=nfft)
			noise[mc,mr] = getpower(ff, np.sqrt(ps), Nfreq, w = 0.1*Nfreq)
	
			if ifplot:
				h,w = 3,1
				xlabels = ['sample  pts', 'sample  pts', 'freq  [Hz]']
				ylabels = ['fb  [pA]', 'fb  [pA]', 'psd  [pA/rtHz]']
				fig, axs = esp.premultps(h, w, 14, 6, lx=xlabels, ly=ylabels)
				fontd = {'fontsize': int(2*min([w*14, h*6]))}
				axs[0].set_title('rowlen=%d col%d row%d'%(rowlen, mc, mr), fontdict = fontd)
				axs[0].plot(y*1e12, color='black', label='r%dc%d, filter:%s'%(mr, mc, str(detrend)))
				zoominx1 = 0
				zoominx2 = int(20*fs) # zoomin to the first 20 s
				axs[0].axvline(x=zoominx1, color='r', linestyle='--')
				axs[0].axvline(x=zoominx2, color='r', linestyle='--',label='zoomin range')
				yz = y[zoominx1:zoominx2]
                                axs[1].plot(yz*1e12, color='black', label='r%dc%d, filter:%s'%(mr, mc, str('linear-for zoomin')))
				axs[2].set_ylim(10, 1e4)
				axs[2].set_xscale('log')
				axs[2].set_yscale('log')
				axs[2].plot(ff, np.sqrt(ps)*1e12, color='black', label='r%dc%d, ps'%(mr, mc))
				pl.axhline(noise[mc,mr]*1e12, color='r', linestyle='--')
				pl.axvline(Nfreq, color='r', linestyle='--')
				esp.posmultps(fig, axs, ffn = outpathps + datafn.split('/')[-1] + '_r%dc%d'%(mr,mc), legloc=1, legfontsize=fontd['fontsize'], ifshow = False)
	return noise, rowlen
			

# calculate the average power
# around target frequency
# within defined bandwindow (default 1Hz)
def getpower(freqarray, psarray, targetf, w = 1):
	if w<2*freqarray[1]:
		w=2*freqarray[1]
	N = len(freqarray)
	return np.mean([psarray[ii] for ii in range(N) if freqarray[ii]<targetf+w*0.5 and freqarray[ii]>targetf-w*0.5])
	#return np.interp(targetf, freqarray, psarray)

def plot_rowlenscan(outputs, cols, rows, Nfreq, outpath):
	for mc in cols:
		for mr in rows:
			nps = outputs[0][:,mc,mr]
			ll  = outputs[1]
			fig, axe = esp.presetting(8,6,'row_len', 'fb [pA/rtHz]')
			axe.set_title('mc%d-mr%d'%(mc,mr))
			axe.scatter(ll, nps*1e12, color='k', marker='+',label='noise current@%.1f Hz'%Nfreq)
			ll_ = np.sort(ll)
			nps_ = np.array([n for _,n in sorted(zip(ll,nps))])
			axe.plot(ll_,nps_*1e12, color='orange', linestyle='--')
			axe.set_ylim(0, 1000)
			esp.possetting(fig, outpath+'mc%dmr%d.png'%(mc,mr), ifshow=False)
	return

if __name__ == "__main__":
	main()
