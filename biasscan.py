import numpy as np
import sys, os
sys.path.insert(0, "/home/cheng/analysis/ploting")
#sys.path.insert(0, "/home/cheng/analysis/timestream")
from tsdata import get_raw_timestream as get_fb
from tsdata import remove_baseline
from lcdata import get_LC, get_PR
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
# usage: python biasscan.py /home/data/cryo/20190508/ noise_vs_bias*
# output in /home/data/output/20190508/biasscan_filt<constant>_nfft<10000>_at<0.1>

def main():
	#options
	#===============================
	Nfreq = 10 
	detrend = 'linear' # for signal.welch
	nfft = 10000	     # for signal.welch
	targetpath = sys.argv[1]
	targetfiles = sys.argv[2]
	calfn = 'calib_BA1'
	cols = range(24)         # the cols in interest. though the data structure will be saved in 24*33 matrix
	rows = range(33)
        count = 25200
        start = 0
        chosenbias=[160,160,200,200,250,260,250,250,
                220,200,200,180,420,420,400,280,
                220,250,280,300,250,260,550,550]
	#************************
	lcpath = '/home/data/cryo/20190912/LC_dark_FPU_260mK_datamode1_runtest2'	
	fitrange = {}
	fitrange["rnal_low"] = 9000
	fitrange["rnal_hgh"] = 10000
	fitrange["rnti_low"] = 900 
	fitrange["rnti_hgh"] = 1000
	fitrange["sc_low"] = 0
	fitrange["sc_hgh"] = 50
	#===============================	

	outpath = targetpath.replace('cryo', 'output') + targetfiles.replace('*','X')  + '/biasscan_filt%s_nfft%d_at%.1f/'%(detrend, nfft, Nfreq)
	if not os.path.isdir(outpath):
		os.makedirs(outpath)

	fnpickle = outpath+'biasscan_Np%.1f.pkl'%Nfreq
	if os.path.isfile(fnpickle):
		ifreproduce = raw_input('Reproduce?')
		shutil.copy2(os.path.realpath(__file__), outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
		if ifreproduce in [1, 'y', 'yes','Y','Yes']:
			reproduce(targetpath, targetfiles, Nfreq,cols,rows,calfn, detrend, nfft, count, start, outpath)
	else:
		
		reproduce(targetpath, targetfiles, Nfreq,cols,rows,calfn, detrend, nfft, count, start, outpath)

	outputs = pickle.load(open(fnpickle,'r'))
	plot_biasscan(outputs, cols, rows, Nfreq, outpath, chosenbias)
	#plot_biasscan_r(outputs, cols, rows, Nfreq, outpath, calfn, lcpath, fitrange)

def reproduce(targetpath, targetfiles, Nfreq,cols,rows, calfn, detrend, nfft, count, start, outpath):
	filelist_ = glob.glob(targetpath+targetfiles)
	filelist = [fn for fn in filelist_ if not (fn.split('.')[-1] == 'run')]
	Np = np.full((len(filelist),24,33), float('nan')) # noise current at certain frequency (Nfreq)
	bias = np.full((len(filelist),24), float('nan'))
        spss = np.full((len(filelist),24,33,nfft/2+1), float('nan'))

	#bar = Bar("files:", max=len(filelist))
	for ii,datafn in enumerate(filelist):
		if datafn.split('.')[-1] == 'run':
			continue
		nn, bb, sps, ff = get_noise(datafn, cols, rows, Nfreq, calfn, detrend, nfft, count, start, ifplot=False, outpathps=outpath+'biasscanps/')
		bias[ii] = bb
		Np[ii] = nn
                spss[ii] = sps
		#bar.next()
	#bar.finish()
	pickle.dump((Np, bias,spss, ff),open(outpath+'biasscan_Np%.1f.pkl'%Nfreq,'w'))

# default: subtract mean, 'constant'
# detrend and nfft will be passed to signal.welch
# outputs: noise (24*33 array), rowlen float
def get_noise(datafn, cols, rows, Nfreq, calfn='calib_BA1', detrend='constant', nfft=10000, count=None, start=0, ifplot=False, outpathps=None):
    print(datafn)
    if not count:
        fbs = get_fb(datafn, calfn=calfn, start=start)
    else:
        fbs = get_fb(datafn, calfn=calfn, count=count, start=start)
	#rowlen = fbs.info['header']['row_len']
	bias = np.full(24, float('nan'))
	for mc in range(24):
		bias[mc] = int(fbs.info['runfile'].data['HEADER']['RB tes bias'].split()[mc])
	fs = fbs.info['freq']
	noise = np.full((24,33), float('nan'))
        sps   = np.full((24,33,nfft/2+1), float('nan'))
	if ifplot and outpathps:
		if not os.path.isdir(outpathps):
			os.makedirs(outpathps)

	for mc in cols:
		for mr in rows:
			y = fbs.fbCalib[mr,mc]
                        bl,yb,mask = remove_baseline(y, n=3)
			ff, ps = signal.welch(yb, fs, return_onesided=True, detrend=detrend, nperseg=nfft)
			noise[mc,mr] = getpower(ff, np.sqrt(ps), Nfreq, w = 0.1*Nfreq)
	                sps[mc,mr]   = np.sqrt(ps)
			if ifplot:
				h,w = 3,1
				xlabels = ['sample  pts', 'sample  pts', 'freq  [Hz]']
				ylabels = ['fb  [pA]', 'fb  [pA]', 'psd  [pA/rtHz]']
				fig, axs = esp.premultps(h, w, 14, 6, lx=xlabels, ly=ylabels)
				fontd = {'fontsize': int(2*min([w*14, h*6]))}
				axs[0].set_title('bias=%d col%d row%d'%(bias[mc], mc, mr), fontdict = fontd)
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
	return  noise, bias, sps, ff
			

# calculate the average power
# around target frequency
# within defined bandwindow (default 1Hz)
def getpower(freqarray, psarray, targetf, w = 1):
	if w<2*freqarray[1]:
		w=2*freqarray[1]
	N = len(freqarray)
	return np.mean([psarray[ii] for ii in range(N) if freqarray[ii]<targetf+w*0.5 and freqarray[ii]>targetf-w*0.5])
	#return np.interp(targetf, freqarray, psarray)

def plot_biasscan(outputs, cols, rows, Nfreq, outpath,chosenbias):
	for mc in cols:
		for mr in rows:
			nps = outputs[0][:,mc,mr]
			bb  = outputs[1][:,mc]
			fig, axe = esp.presetting(8,6,'bias, [ADU]', 'fb [pA/rtHz]')
			axe.set_title('mc%d-mr%d'%(mc,mr))
			axe.scatter(bb, nps*1e12, color='k', marker='+',label='noise current@%.1f Hz'%Nfreq)
			bb_ = np.sort(bb)
			nps_ = np.array([n for _,n in sorted(zip(bb,nps))])
			axe.plot(bb_,nps_*1e12, color='orange', linestyle='--')
                        if not np.isnan(chosenbias[mc]):
                            axe.axvline(x=chosenbias[mc], color='r')
			axe.set_ylim(0,350)
			esp.possetting(fig, outpath+'mc%dmr%d.png'%(mc,mr), ifshow=False)
	return

# refer bias points to resistances
def plot_biasscan_r(outputs, cols, rows, Nfreq, outpath, calfn=None, lcpath=None, fitrange=None):

	if lcpath and calfn:
		lcfn = lcpath.split('/')[-1]
		flc = mce_data.MCEFile(lcpath+'/'+lcfn)
		blc = np.loadtxt(lcpath+'/'+lcfn+'.bias',skiprows=1)
		ylc = -1.0*flc.Read(row_col=True,unfilter='DC').data

	for mc in cols:
		for mr in rows:
			nps = outputs[0][:,mc,mr]
			bb  = outputs[1][:,mc]
			bb_ = np.sort(bb)
			nps_ = np.array([n for _,n in sorted(zip(bb,nps))])
			if lcpath and calfn:
				if calfn == 'calib_BA1':
					calib = calib_BA1.calib_ba1()
				elif calfn == 'calib_SK':
					calib = calib_SK.calib_sk()
				else:
					print('!!! wrong calfn. default BA1')
					calib = calib_BA1.calib_ba1()
				biascalib, fbcalib, rnti, ksc = get_LC(blc, ylc[mr,mc], calib, fitrange)
				rr, pp, psat = get_PR(biascalib, fbcalib, calib, rnti, rnti*0.5)
				bb = np.interp(bb, blc[::-1], rr[::-1])
				bb_ = np.interp(bb_, blc[::-1], rr[::-1])
				fig, axe = esp.presetting(8,6,r'$R_{opt}, [m\Omega]$', 'fb [pA/rtHz]')
				axe.axvline(x = rnti*1000, color='red', label=r'$R_n$')
			else:
				fig, axe = esp.presetting(8,6,'bias, [ADU]', 'fb [pA/rtHz]')
			axe.set_title('mc%d-mr%d'%(mc,mr))
			axe.scatter(bb*1000, nps*1e12, color='k', marker='+',label='noise current@%.1f Hz'%Nfreq)
			axe.plot(bb_*1000,nps_*1e12, color='orange', linestyle='--')
			axe.set_ylim(0,350)
			esp.possetting(fig, outpath+'r_mc%dmr%d.png'%(mc,mr), ifshow=False)
	return

if __name__ == "__main__":
	main()
