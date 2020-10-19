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
	targetpath = sys.argv[1]
	cols = range(24)         # the cols in interest. though the data structure will be saved in 24*33 matrix
	rows = range(33)
	#===============================	

        #calculate err
	outpath = targetpath.replace('cryo', 'output') + 'rowlenscan/'
	if not os.path.isdir(outpath):
		os.makedirs(outpath)
	fnpickle = outpath+'rowlenscan.pkl'
	if os.path.isfile(fnpickle):
		ifreproduce = raw_input('Reproduce?')
		shutil.copy2(os.path.realpath(__file__), outpath + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
		if ifreproduce in [1, 'y', 'yes','Y','Yes']:
			reproduce(targetpath, cols, rows,outpath)
	else:
		reproduce(targetpath, cols, rows, outpath)

	outputs = pickle.load(open(fnpickle,'r'))
	plot_rowlenscan(outputs, cols, rows, outpath)

def reproduce(targetpath,cols,rows,outpath):
	filelistERR_= glob.glob(targetpath+'*ERR')
	filelistERR = [fn for fn in filelistERR_ if not (fn.split('.')[-1] == 'run')]
	rowlens = np.full(len(filelistERR), float('nan')) 
	NpERR = np.full((len(filelistERR),24,33), float('nan'))
	NpFB1 = np.full((len(filelistERR),24,33), float('nan'))
	NpFB10= np.full((len(filelistERR),24,33), float('nan'))
        print(filelistERR)
	for ii,datafn in enumerate(filelistERR):
		if datafn.split('.')[-1] == 'run':
			continue
                fbsERR = get_fb(datafn, calfn='calib_BA1')
                fbsFB1 = get_fb(datafn.replace('ERR','FB1'), calfn='calib_BA1')
                fbsFB10= get_fb(datafn.replace('ERR','FB10'), calfn='calib_BA1')
                
                for col in cols:
                    for row in rows:
                        NpERR[ii,col,row] = np.std(fbsERR.fb[row,col])
                        NpFB1[ii,col,row] = np.std(fbsFB1.fbCalib[row,col])
                        NpFB10[ii,col,row] = np.std(fbsFB10.fbCalib[row,col])
		rowlens[ii] = fbsERR.info['header']['row_len']
	pickle.dump((NpERR, NpFB1, NpFB10, rowlens),open(outpath+'rowlenscan.pkl','w'))

def plot_rowlenscan(outputs, cols, rows,  outpath):
	for mc in cols:
		fig, axs = esp.premultps(3,1,10.,4.,['row_len','row_len','row_len'], 
                            [r'ERR <$\sigma$> ADU', r'FB1 <$\sigma$> pA', r'FB10 <$\sigma$> pA'])
		ll  = outputs[3]
		ll_ = np.sort(ll)
		for mr in rows:
			npsERR = outputs[0][:,mc,mr]
                        npsFB1 = outputs[1][:,mc,mr]
                        npsFB10= outputs[2][:,mc,mr]
			npsERR_ = np.array([npsERR[np.where(ll==ll_[i])[0][0]] for i in range(len(ll))])
			npsFB1_ = np.array([npsFB1[np.where(ll==ll_[i])[0][0]] for i in range(len(ll))])
			npsFB10_= np.array([npsFB10[np.where(ll==ll_[i])[0][0]] for i in range(len(ll))])
                        axs[0].plot(ll_,npsERR_)
                        axs[1].plot(ll_,npsFB1_*1e12)
                        axs[2].plot(ll_,npsFB10_*1e12)
                        axs[0].set_yscale('log')
                        axs[0].grid()
                        axs[1].set_yscale('log')
                        axs[1].grid()
                        axs[2].set_yscale('log')
		axs[0].set_title('col %d'%(mc))
		esp.possetting(fig, outpath+'rowlenscan_col%d.png'%(mc), ifshow=False)
	return

if __name__ == "__main__":
	main()
