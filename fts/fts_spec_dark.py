import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sio
from scipy.fftpack import dct,dst,fft
import numpy as np
import math
import glob
import sys, os
import os.path
import pylab as pl
import mce_data
sys.path.insert(0, "/home/cheng/Pole2019/analysispy/tespy")
from lcdata import get_PLC
import calib_SK
import calib_BA1
import cPickle as pickle
import ba30_ModuleMapping_N3 as minfo
import sys,os
sys.path.insert(0, "/home/cheng/Pole2019/analysispy/tespy/fts/filelist")
from ba30_SK_N7N6_v2 import flist

testbed = 'SK'
if testbed == 'SK':
	calib = calib_SK.calib_sk()
elif testbed == 'BA':
	calib = calib_BA1.calib_ba1()
else:
	calib = None

fitrange = {	'rnti_low': 30000.00,
		'rnti_hgh': 35000.00,
		'sc_low': None,
		'sc_hgh': None}

c = 299792458.
v_mirror = 2.0e-3 #mm/s for the harvard fts, v_mirror = 1.905e-3 #mm/s
nSample = 210.44 #Mux-11, row_len-120, datarate-60, row_num-33
delta_step = 2*v_mirror/nSample
fNyq = c/2/delta_step/1e9 #in GHz
nlen_hlf_intf = 13000 # number of points to use on each side of the wlf
highp = 50 # set the x limit of the spectrum plot
lowp = 0 
kB = 1.38e-11 # match pW
small = False 
doEff = 0 
date  = '20210511'
runn = 'SK_N7B6' # N7B6 is typo for N7N6.
cols = [8,9,0,1]
rows = range(33)
inftsl = '/home/data/output/%s/%s_fts_perfn.pkl'%(date,runn)
indpdt = '/home/data/output/20210423/20210423_SK_BA30N6N7/20210423_SK_BA30N6N7_dpdt_rnti.pkl'

def get_darklist(fnplc,rntis,pols,wlfs):
# use plc data to decide which dark detectors are in 
# transition. 

	wlf_count = np.where(np.isnan(wlfs),0,1)
	nwlf = np.sum(wlf_count,(1,2))
	print 

	plc = get_PLC(fnplc,fitrange,calib,-1)
	darklist = []
	for col in cols:
		# make sure there is signal in this tile
		if nwlf[col]==0 and nwlf[col + (-1)**np.mod(col,2)]==0:
			continue
		for row in rows:
			if not pols[col][row]=='D':
				continue
			# check if in transition. Al normal is quite unlikely  
			if np.isnan(rntis[col][row]) or rntis[col][row]<=0:
				continue
			if np.isnan(plc.R0[row,col]) or plc.R0[row,col]<=0:
				continue
			if plc.R0[row,col]*1000<=rntis[col][row] or plc.R0[row,col]>=plc.R[row,col]*0.95:
				continue
			darklist.append((col,row))

	return darklist

def main():

	# load outputs from fts_spec_v2_perfn.py
	dftsl = pickle.load(open(inftsl,'r'))
	detcol_array = dftsl[4]
	detrow_array = dftsl[5]
	detpol_array = dftsl[6]
	wlf_array = dftsl[10]
	
	# load dpdt data
	ddpdt = pickle.load(open(indpdt,'r'))
	rnti_array = ddpdt[1]

	outdir='/home/data/output/%s'%date
	if not os.path.isdir(outdir):
		os.makedirs(outdir)
	fnpickle='%s/%s_dark.pkl'%(outdir,runn)

	outfigdir = '%s/spec_dark_perfn'%outdir;
	if not os.path.isdir(outfigdir):
		os.makedirs(outfigdir)

	x = np.array(range(2*nlen_hlf_intf+1))	
	xp = x*1/nSample*v_mirror # mm

	if os.path.exists(fnpickle):
		d = pickle.load(open(fnpickle,'r')) 
		filelist = d[0]
		fb_hs_final = d[1]
		cs_array = d[2]
		ss_array = d[3]
		freq = d[4]

	else:
		x = np.array(range(2*nlen_hlf_intf+1))
	
		# collect relevant scans
		filelist = flist.fns
		fb_hs_array = np.full((24,33,len(filelist),10,2*nlen_hlf_intf+1), np.nan)
		for ifn,fn in enumerate(filelist):
			print "Check file (%d/%d): %s"%(ifn+1,len(filelist),fn)
	
			fnplcdir = '/home/data/cryo/%s'%fn[:8]
			fnplcs = np.sort(glob.glob('%s/*_plc_bias_al'%fnplcdir))
			fnplc = fnplcs[-1]
			darklist = get_darklist(fnplc, rnti_array, detpol_array, wlf_array[:,:,ifn,:])
			if len(darklist)==0:
				continue
			wlfdd = np.nanmedian(wlf_array[:,:,ifn,:],(0,1))
			f = mce_data.MCEFile('/home/data/cryo/%s/%s'%(fn[:8],fn))
			fb_all = f.Read(row_col=True,unfilter='DC').data

			print fn
			print darklist
	
			for idd,dd in enumerate(darklist):			
				fb = fb_all[dd[1],dd[0]]
	
				for iwlf,wlf in enumerate(wlfdd):
					if np.isnan(wlf):
						continue
					wlf = int(wlf)
					fb_hs = fb[wlf-nlen_hlf_intf:wlf+nlen_hlf_intf+1]
					if len(fb_hs) != len(x):
						continue
					polyb = np.poly1d(np.polyfit(x, fb_hs, 3))
					fb_hs = fb_hs - polyb(x)
					fb_hs_array[dd[0],dd[1],ifn,iwlf,:]=fb_hs
		
		# get spectra
		freq_ = np.linspace(1,fNyq,nlen_hlf_intf+1)
		freq = freq_[:-1]
	
		fb_hs_final = np.full((24,33,2*nlen_hlf_intf+1), np.nan)
	
		#fft_spec,cs,ss = get_best_symm(fb_hs_final, -nlen_hlf_intf, 0)
		nroll = -nlen_hlf_intf
		cs_array = np.full((24,33,nlen_hlf_intf),np.nan)
		ss_array = np.full((24,33,nlen_hlf_intf),np.nan)
		for col in cols:
			for row in rows:
				fb_hs_array1 = fb_hs_array[col,row] 
				if np.all(np.isnan(fb_hs_array1)):
					continue
				for ifn in range(len(filelist)):
					if np.all(np.isnan(fb_hs_array1[ifn])):
						continue
					fb_hs=np.nanmean(fb_hs_array1[ifn,:],0)
					fft_spec = fft(np.roll(fb_hs,nroll))
					cs = fft_spec[:nlen_hlf_intf].real
					ss = fft_spec[:nlen_hlf_intf].imag
	
					pl.figure(figsize=(18,15), dpi=80)
		
					ax = pl.subplot(2,1,1)
					plt.plot(xp*1.0e3,fb_hs,color='blue')
					plt.text(0.03,0.9,'mr'+str(int(row))+' mc'+str(int(col))+' D',fontsize=25,transform=ax.transAxes)
					plt.text(0.03,0.85,filelist[ifn],fontsize=25,transform=ax.transAxes)
					plt.xlabel('mm',fontsize=25)
					plt.ylabel('fb',fontsize=25)
					plt.xlim(0,(2*nlen_hlf_intf+1)/nSample*(v_mirror*1.0e3))
					plt.ylim(-2./3*(max(fb_hs)-min(fb_hs)),
						2./3*(max(fb_hs)-min(fb_hs)))
					for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
							ax.get_xticklabels() + ax.get_yticklabels()):
						item.set_fontsize(20)
					
					ax = pl.subplot(2,1,2)
					plt.plot(freq,cs,'r',freq,ss,'g')
					plt.xlim(0,1000)
					plt.xlabel('freq GHz',fontsize=25)
					
					figname = '%s/spec_row%d_col%d_ifn%d.png'%(outfigdir,row,col,ifn)
					print figname
					pl.savefig(figname)

					
				fb_hs_final[col,row] = np.nanmean(fb_hs_array1,(0,1))
				fft_spec = fft(np.roll(fb_hs_final[col,row],nroll))
				cs_array[col,row] = fft_spec[:nlen_hlf_intf].real
				ss_array[col,row] = fft_spec[:nlen_hlf_intf].imag
	
				#plt.plot(freq,cs_array[col,row])
				#plt.show()	
	
		pickle.dump((filelist,fb_hs_final,cs_array,ss_array,freq),open(fnpickle,'w'))	

	# make plots
	outfigdir = '%s/spec_dark'%outdir;
	if not os.path.isdir(outfigdir):
		os.makedirs(outfigdir)

	for col in cols:
		for row in rows:
			if np.all(np.isnan(cs_array[col,row])):
				continue
			fb_hs = fb_hs_final[col,row]
			cs = cs_array[col,row]
			ss = ss_array[col,row]

			pl.figure(figsize=(18,15), dpi=80)

			ax = pl.subplot(2,1,1)
			plt.plot(xp*1.0e3,fb_hs,color='blue')
			plt.text(0.03,0.9,'mr'+str(int(row))+' mc'+str(int(col))+' D',fontsize=25,transform=ax.transAxes)
			plt.xlabel('mm',fontsize=25)
			plt.ylabel('fb',fontsize=25)
			plt.xlim(0,(2*nlen_hlf_intf+1)/nSample*(v_mirror*1.0e3))
			plt.ylim(-2./3*(max(fb_hs)-min(fb_hs)),
				2./3*(max(fb_hs)-min(fb_hs)))
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
					ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(20)
			
			ax = pl.subplot(2,1,2)
			plt.plot(freq,cs,'r',freq,ss,'g')
			plt.xlim(lowp,highp)
			plt.xlabel('freq GHz',fontsize=25)
			
			figname = '%s/spec_row%d_col%d_freqrange%d.png'%(outfigdir,row,col,highp)
			print figname
			#plt.show()
			pl.savefig(figname)

if __name__=='__main__':
    main()
