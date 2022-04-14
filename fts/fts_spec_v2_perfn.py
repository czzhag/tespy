import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sio
from scipy.fftpack import dct,dst,fft
import numpy as np
import math
import sys, os
import os.path
import pylab as pl
import mce_data
sys.path.insert(0, "../")
import calib_SK
import calib_BA1
import cPickle as pickle
import ba30_ModuleMapping_N3 as minfo
import sys,os
sys.path.insert(0, "./filelist")
from ba30_SK_N7N6_v2 import flist

testbed = 'SK'

c = 299792458.
v_mirror = 2.0e-3 #mm/s for the harvard fts, v_mirror = 1.905e-3 #mm/s
nSample = 210.44 #Mux-11, row_len-120, datarate-60, row_num-33
delta_step = 2*v_mirror/nSample
fNyq = c/2/delta_step/1e9 #in GHz
nlen_hlf_intf = 13000 # number of points to use on each side of the wlf
highp = 50 # set the x limit of the spectrum plot
lowp = 0 
kB = 1.38e-11 # match pW
small = False # make small plots
savefigdetcoord=1  
date  = '20210511'
runn = 'SK_N7B6_perfn'
cols = [8,9,0,1] # mce cols to run through. mce row is default as 0-32
doEff = 1 # if you want to scale the y axis of the spectrum to optical efficiency
indpdt = '/home/cheng/analysis/DATA/output/20210423/20210423_SK_BA30N6N7/20210423_SK_BA30N6N7_dpdt_rnti.pkl' # specify the dpdt data file to use if you set doEff=1

def mask_spikes(fb, nfb, hwin=15):
	mask = np.full(len(fb), 1)
	for ii in range(len(fb)):
		try:
			ts = fb[ii-hwin:ii+hwin]
		except:
			continue
		std = np.std(ts)
		if std>200:
			#print(std)
			mask[ii-hwin:ii+hwin] = 0
	fb = fb*mask
	nfb = nfb+mask
	return fb, nfb
			
def dataNwlf():
	
	filelist = flist.fns
        chnlist  = flist.chns	
	wlf = flist.wlf
	return filelist,chnlist,wlf

def adjust_wlf(interf,guess_tolerance=25):
	#===================================================================
	# This func is using the similar strategy as grant did in the MATLAB version
	# (getting max corr between left and right of wlf)
	# I basically just translate it into python.
	#
	# Though the output here is the movement need to be made on the first 
	# guessed wlf. 
	#
	# Cheng 07/30/18
	#===================================================================

	interf=scipy.signal.savgol_filter(interf,55,5)
	# Truncate the last data point if the input is even.
	if len(interf)%2==0:
		interf = interf[:-1]
	
	N = len(interf)

	# Consider guess_tolerance indices to the left and right of the center.
	wlf_guesses = np.linspace(-guess_tolerance+(N-1)/2,guess_tolerance+(N-1)/2,2*guess_tolerance+1)
	half_length = int(math.floor((N-guess_tolerance*4-1)/2))
	correlations = [0 for i in range(2*guess_tolerance+1)]
	
	for jj in range(2*guess_tolerance+1):
		# Find the indices just above and below the wlf guess.
		below_wlf = int(math.ceil(wlf_guesses[jj]-0.1)-1)
		above_wlf = int(math.ceil(wlf_guesses[jj]+0.1))
		# Separate the data into the two halves, excluding the wlf and anything
		# on the tail end longer than half_length.
		left_data = interf[-half_length+below_wlf:below_wlf]
		right_data = interf[above_wlf:half_length+above_wlf]
		#plt.plot(np.linspace(0,len(left_data),len(left_data)),left_data)
		#plt.plot(len(left_data)+np.linspace(0,len(right_data),len(right_data)),right_data)
		#plt.show()
		# See how well the two halves correlate.
		correlations[jj] = np.corrcoef(left_data,right_data[::-1])[0,1]
		
	Dwlf = wlf_guesses[np.argmax(correlations)]-(N-1)/2
	#plt.plot(correlations)
	#plt.show()
	return Dwlf
	
def get_bcbw(cs,freq):
	
	N = len(cs)
	dv = freq[1]-freq[0]
	S = 0
	SS = 0
	vS = 0
	for ii in range(N-1):
		if freq[ii]>highp or freq[ii]<lowp:
			continue
		S += cs[ii]*dv
		SS += cs[ii]**2*dv
		vS += cs[ii]*(freq[ii]+freq[ii+1])/2.*dv
		
		
	bc = vS/S
	bw = S**2/SS
	
	return bc,bw

def get_best_symm(fb_hs_final, nroll, ntry=10, output_dwlf = False):
	
	mss = np.full(ntry*2, float("nan"))
	for ii in range(ntry):
		for sign in [0,1]:
			move = ii*2*(-1)**sign
			fft_spec = fft(np.roll(fb_hs_final,nroll+move))
			cs = np.zeros(nlen_hlf_intf)
			ss = np.zeros(nlen_hlf_intf)
			cs = fft_spec[:nlen_hlf_intf].real
			ss = fft_spec[:nlen_hlf_intf].imag
			mss[ii*2+sign] = max(np.abs(ss))
	
	n = np.argmin(mss)
	move = (n-np.mod(n,2))*(-1)**np.mod(n,2)
	if output_dwlf:
		return move
	fft_spec = fft(np.roll(fb_hs_final,nroll+move))
	cs = np.zeros(nlen_hlf_intf)
	ss = np.zeros(nlen_hlf_intf)
	cs = fft_spec[:nlen_hlf_intf].real
	ss = fft_spec[:nlen_hlf_intf].imag
	
	return fft_spec, cs, ss

	
def y2eff(cs,ss,freq,dpdt):
	
	N = len(cs)
	S = 0
	dv = (freq[1]-freq[0])*1.0e9
	for ii in range(N-1):
		if freq[ii]>highp or freq[ii]<lowp:
			continue
		S += cs[ii]*dv
	effc = dpdt*cs/S/kB
	effs = dpdt*ss/S/kB
	
	return effc,effs
	
def main():
	
	#===================================================================
	# input data
	#===================================================================
	in_path  = '/home/cheng/analysis/DATA/cryo/'
	filelist,chnlist,wlf = dataNwlf()
	out_path_main = '/home/cheng/analysis/DATA/output/%s/'%date
	if small:
		out_fig = out_path_main+'spec_%d_perfn/'%highp
		if not os.path.isdir(out_fig):
			os.makedirs(out_fig)
	else:
		out_fig = out_path_main+'spec_large_%d_perfn/'%highp
		if not os.path.isdir(out_fig):
			os.makedirs(out_fig)
	bc_array = np.full((24,33,len(filelist)),np.nan)
	bw_array = np.full((24,33,len(filelist)),np.nan)
	detpol_array = [[None for i in range(33)] for j in range(24)]
	eff_array = np.full((24,33,len(filelist)),np.nan)
	detcol_array = np.full((24,33),np.nan)
	detrow_array = np.full((24,33),np.nan)
	wlf_array = np.full((24,33,len(filelist),10),np.nan)
	interfg = {}
	spcos = {}
	spsin = {}

	fnpickle = out_path_main + '%s_fts_perfn.pkl'%runn

	if os.path.exists(fnpickle):
		dexist=pickle.load(open(fnpickle,'r')) 
		bw_array=dexist[1] 
		bc_array=dexist[2]
		eff_array=dexist[3]
		detcol_array=dexist[4]
		detrow_array=dexist[5]
		detpol_array=dexist[6]
		interfg = dexist[7]
		spcos = dexist[8]
		spsin = dexist[9]	
		wlf_array = dexist[10]

	if os.path.exists(indpdt):
		d2 = pickle.load(open(indpdt,'r'))
	
	for ifn,filename in enumerate(filelist):
	
		f = mce_data.MCEFile(in_path+'%s/'%filename[0:8]+filename)
		dname = os.path.split(in_path+filename)[0]
		fb_all = f.Read(row_col=True,unfilter='DC').data
	
		for col in cols:
			for row in range(33):
				
				if not ('r'+str(int(row))+'c'+str(int(col)) in chnlist[filename]):
					continue
	
				if os.path.exists(indpdt):
					dpdt = d2[0][col][row]
				else:
					print('NO DPDT data!')
					exit
			
				#===================================================================
				# get the interferogram
				# wlf at center, each side has 'nlen_hlf_intf' data points (not counting wlf)
				# 
				# NOTE: the input wlfs form 'dataNwlf' are not necessary to be precise
				#		the 'adjust_wlf(interf,guess_tolerance)' function will deal with it.
				#===================================================================
				
				x = np.array(range(2*nlen_hlf_intf+1))	
				xp = x*1/nSample*v_mirror # mm
	
				im, detcol,detrow,detpol = minfo.mce2det(col,row)
				detcol_array[col][row] = detcol
				detrow_array[col][row] = detrow
				detpol_array[col][row] = detpol
				
	
				fb_hs_list = []
				nhs = np.zeros(2*nlen_hlf_intf+1)
				Nhs = 0
				fb_hs_final = np.array([0]*(2*nlen_hlf_intf+1))
	
				fb = fb_all[row,col]
	
				wlfs = wlf[filename]
				for iwlf,wlf_prim in enumerate(wlfs):
					fb_hs = fb[wlf_prim-9000:wlf_prim+9000+1]
					x0=np.linspace(0,len(fb_hs),len(fb_hs))
					polyb = np.poly1d(np.polyfit(x0, fb_hs, 3))
					fb_hs = fb_hs - polyb(x0)
					wlf_better = int(round(adjust_wlf(fb_hs,55)))+wlf_prim
					wlf_array[col][row][ifn][iwlf] = wlf_better
					fb_hs = fb[wlf_better-nlen_hlf_intf:wlf_better+nlen_hlf_intf+1]
					if len(fb_hs)<len(x):
						continue
					polyb = np.poly1d(np.polyfit(x, fb_hs, 3))
					fb_hs = fb_hs - polyb(x)
					fb_hs, nhs = mask_spikes(fb_hs, nhs)
					fb_hs_final =  fb_hs_final + fb_hs
					fb_hs_list.append(fb_hs)
					Nhs += 1
	
	
				if not Nhs:
					continue
				
				# plot
				if small:
					pl.figure(figsize=(10,8), dpi=20)
				else:
					pl.figure(figsize=(18,15), dpi=80)
				ax = pl.subplot(2,1,1)
				for i in range(Nhs):
					plt.plot(xp*1.0e3,fb_hs_list[i],color='grey',alpha=0.2)
				fb_hs_final = fb_hs_final/Nhs
				plt.plot(xp*1.0e3,fb_hs_final,color='blue')
				plt.text(0.03,0.9,'mr'+str(int(row))+' mc'+str(int(col))+', dr'+str(int(detrow))+' dc'+str(int(detcol))+' pol'+detpol,fontsize=25,transform=ax.transAxes)
				plt.xlabel('mm',fontsize=25)
				plt.ylabel('fb',fontsize=25)
				plt.xlim(0,(2*nlen_hlf_intf+1)/nSample*(v_mirror*1.0e3))
				plt.ylim(-2./3*(max(fb_hs_final)-min(fb_hs_final)),
					2./3*(max(fb_hs_final)-min(fb_hs_final)))
				for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
						ax.get_xticklabels() + ax.get_yticklabels()):
					item.set_fontsize(20)
				
				#===================================================================
				# get frequency
				#===================================================================
				freq_ = np.linspace(1,fNyq,nlen_hlf_intf+1)
				freq = freq_[:-1]
				
				#===================================================================
				# get spectra -- still not 100% sure this part is right.
				#===================================================================
				fft_spec,cs,ss = get_best_symm(fb_hs_final, -nlen_hlf_intf)
				bc,bw = get_bcbw(cs,freq)
				if doEff:
					cs,ss = y2eff(cs,ss,freq,dpdt)
					avereff = dpdt/kB/(bw*1.0e9)
				else:
					avereff = 0	
				# plot
				ax = pl.subplot(2,1,2)
				plt.plot(freq,cs,'r',freq,ss,'g')
				plt.xlim(lowp,highp)
				plt.xlabel('freq GHz',fontsize=25)
				if doEff:
					plt.ylim(-0.05,1.0)
					plt.ylabel('efficiency',fontsize=25)
					plt.text(0.03,0.8,'aver eff = '+str(round(avereff*100,0))+'%',fontsize=25,transform=ax.transAxes)
				for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
						ax.get_xticklabels() + ax.get_yticklabels()):
					item.set_fontsize(20)
				plt.text(0.03,0.9,'bc = '+str(round(bc,2))+'GHz, bw = '+str(round(bw/bc*100,1))+'%',fontsize=25,transform=ax.transAxes)
		
				if not savefigdetcoord:
					figname = 'spec_row'+str(int(row))+'_col'+str(int(col))+'_fn'+str(int(ifn))
				else:
					figname = 'spec_im'+str(int(im))+'_detrow'+str(int(detrow))+'_detcol'+str(int(detcol))+'_'+detpol+'_fn'+str(int(ifn))
				#plt.show()	
				fn = os.path.join(out_fig, figname+'.png')
				pl.savefig(fn)
			
				bw_array[col][row][ifn] = bw/bc*100
				bc_array[col][row][ifn] = bc
				eff_array[col][row][ifn] = avereff*100
				interfg['r%dc%df%d'%(row,col,ifn)] = fb_hs_final
				spcos['r%dc%df%d'%(row,col,ifn)] = cs
				spsin['r%dc%df%d'%(row,col,ifn)] = ss		
	
	pickle.dump((filelist, bw_array, bc_array, eff_array, detcol_array, detrow_array, detpol_array, interfg, spcos, spsin, wlf_array, v_mirror, nSample),open(fnpickle,'w'))
	
	
if __name__=='__main__':
    main()
