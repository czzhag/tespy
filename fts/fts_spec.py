import matplotlib.pyplot as plt
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
#import ba40_ModuleMapping_BA as ba40_ModuleMapping
import ba30_ModuleMapping as minfo
import sys,os
sys.path.insert(0, "./filelist")
from ba30_SK_N5_B_highsq1 import flist
#from ba30_SK_N1 import flist
#from ba40_T3 import flist
#import despike

# not really need to match the testbed you used to take data, but can be used to specify the 
# style you used in the filelist
testbed = 'SK'

c = 299792458.
#v_mirror = 1.905e-3 #mm/s
v_mirror = 2.0e-3 #mm/s
nSample = 210.44 #Mux-11, row_len-120, datarate-60, row_num-33
delta_step = 2*v_mirror/nSample
fNyq = c/2/delta_step/1e9 #in GHz
nlen_hlf_intf = 13000
nlen_fb = nlen_hlf_intf+1000
highp = 1000
lowp = 20
kB = 1.38e-11 # match pW
small = True 
doEff = 0
date  = 20201020
#date = 20201022
#date = 20191106
#date = 20181210
runn = 'SK_N5_B_highsq1'
#runn = 'SK_N5_B_lowsq1'
#runn = 'SK_N1'
#runn = 'SK_T3'
cols = [0,1]
indpdt = '/home/cheng/analysis/DATA/output/20191106/BA30N1-SK/BA30N1-SK_dpdt_rnti.pkl'

def mask_spikes(fb, nfb, hwin=15):
	mask = np.full(len(fb), 1)
	for ii in range(len(fb)):
		try:
			ts = fb[ii-hwin:ii+hwin]
		except:
			continue
		std = np.std(ts)
		if std>22:
			#print(std)
			mask[ii-hwin:ii+hwin] = 0
	fb = fb*mask
	nfb = nfb+mask
	return fb, nfb
			
def dataNwlf():
	
	filelist = flist.fns	
	wlf = flist.wlf
	#cwlf = flist.commonwlf
	return filelist,wlf

def dataNwlfBA():
	filelist = flist.fns
	wlf = flist.commonwlf
	return filelist,wlf

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
	
	# Truncate the last data point if the input is even.
	if len(interf)%2==0:
		interf = interf[:-1]
	
	N = len(interf)

	# Consider guess_tolerance indices to the left and right of the center.
	wlf_guesses = np.linspace(-guess_tolerance+(N-1)/2,guess_tolerance+(N-1)/2,4*guess_tolerance+1)
	half_length = int(math.ceil(wlf_guesses[0])-1)
	correlations = [0 for i in range(4*guess_tolerance+1)]
	
	for jj in range(4*guess_tolerance+1):
		# Find the indices just above and below the wlf guess.
		below_wlf = int(math.ceil(wlf_guesses[jj]-0.1)-1)
		above_wlf = int(math.ceil(wlf_guesses[jj]+0.1))
		# Separate the data into the two halves, excluding the wlf and anything
		# on the tail end longer than half_length.
		left_data = interf[-half_length+below_wlf:below_wlf]
		right_data = interf[above_wlf:half_length+above_wlf]
		# See how well the two halves correlate.
		correlations[jj] = np.corrcoef(left_data,right_data)[0,1]
		
	Dwlf = wlf_guesses[np.argmax(correlations)]-(N-1)/2
	
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
	in_path  = '/home/cheng/analysis/DATA/cryo/%s/'%date
	if testbed == 'BA':
		filelist,wlf = dataNwlfBA()
	else:
		filelist,wlf = dataNwlf()
	out_path_main = '/home/cheng/analysis/DATA/output/%s/'%date
	if small:
		out_fig = out_path_main+'spec/'
		if not os.path.isdir(out_fig):
			os.makedirs(out_fig)
	else:
		out_fig = out_path_main+'spec_large/'
		if not os.path.isdir(out_fig):
			os.makedirs(out_fig)
	bc_array = [[0 for i in range(33)] for j in range(24)]
	bw_array = [[0 for i in range(33)] for j in range(24)]
	eff_array = [[0 for i in range(33)] for j in range(24)]
	detcol_array = [[0 for i in range(33)] for j in range(24)]
	detrow_array = [[0 for i in range(33)] for j in range(24)]
	detpol_array = [[0 for i in range(33)] for j in range(24)]
	interfg = {}
	spcos = {}
	spsin = {}

	for col in cols:
		for row in range(33):
			if not ('r'+str(row)+'c'+str(col) in filelist):
				continue
			if not (testbed == 'BA'):
				if not ('r'+str(row)+'c'+str(col) in wlf):
					continue
			
			if os.path.exists(indpdt):
				d2 = pickle.load(open(indpdt,'r'))
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
			
			fb_hs_final = np.array([0]*(2*nlen_hlf_intf+1))
			x = np.array(range(2*nlen_hlf_intf+1))	
			xp = x*1/nSample*v_mirror # mm
			
			fb_hs_list = []
			nhs = np.zeros(2*nlen_hlf_intf+1)
			Nhs = 0
			for filename in filelist['r'+str(int(row))+'c'+str(int(col))]:
				if not (testbed == 'BA'):	
					if not (filename in wlf['r'+str(int(row))+'c'+str(int(col))]):
						continue
				f = mce_data.MCEFile(in_path+filename)
				dname = os.path.split(in_path+filename)[0]
				fb_all = f.Read(row_col=True,unfilter='DC').data
				fb = fb_all[row,col]
				im, detcol,detrow,detpol = minfo.mce2det(col,row)
				if not (testbed == 'BA'):
					wlfs = wlf['r'+str(int(row))+'c'+str(int(col))][filename]
				else:
					wlfs = wlf
				for wlf_prim in wlfs:
					fb_hs = fb[wlf_prim-nlen_hlf_intf:wlf_prim+nlen_hlf_intf+1]
					wlf_better = int(round(adjust_wlf(fb_hs,30)))+wlf_prim
					#wlf_better = -get_best_symm(fb_hs, -nlen_hlf_intf, ntry=10, output_dwlf=True) + wlf_prim	
					fb_hs = fb[wlf_better-nlen_hlf_intf:wlf_better+nlen_hlf_intf+1]
					if len(fb_hs)<len(x):
						continue
					polyb = np.poly1d(np.polyfit(x, fb_hs, 3))
					fb_hs = fb_hs - polyb(x)
					fb_hs, nhs = mask_spikes(fb_hs, nhs)
					fb_hs_final =  fb_hs_final + fb_hs
					fb_hs_list.append(fb_hs)
					Nhs += 1
		
			# plot
			if small:
				pl.figure(figsize=(10,8), dpi=20)
			else:
				pl.figure(figsize=(18,15), dpi=80)
			ax = pl.subplot(2,1,1)
			for i in range(Nhs):
				plt.plot(xp*1.0e3,fb_hs_list[i],color='grey',alpha=0.2)
			fb_hs_final = fb_hs_final/nhs
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
			#print(fb_hs_final)
			fft_spec,cs,ss = get_best_symm(fb_hs_final, -nlen_hlf_intf)
			'''
			fft_spec = fft(np.roll(fb_hs_final,-nlen_hlf_intf))
			cs = np.zeros(nlen_hlf_intf)
			ss = np.zeros(nlen_hlf_intf)
			
			cs = fft_spec[:nlen_hlf_intf].real
			ss = fft_spec[:nlen_hlf_intf].imag
			'''
			#for i in range(1,nlen_hlf_intf):
				#cs[i] = np.real(fft_spec[i]+np.conjugate(fft_spec[-i]))
				#ss[i] = np.imag(-fft_spec[i]+np.conjugate(fft_spec[-i]))
			bc,bw = get_bcbw(cs,freq)
			if doEff:
				cs,ss = y2eff(cs,ss,freq,dpdt)
				avereff = dpdt/kB/(bw*1.0e9)
			else:
				avereff = 0	
			# plot
			ax = pl.subplot(2,1,2)
			plt.plot(freq,-cs,'r',freq,ss,'g')
			plt.xlim(lowp,highp)
			plt.xlabel('freq GHz',fontsize=25)
			if doEff:
				plt.ylabel('efficiency',fontsize=25)
				plt.text(0.03,0.8,'aver eff = '+str(round(avereff*100,0))+'%',fontsize=25,transform=ax.transAxes)
			for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
					ax.get_xticklabels() + ax.get_yticklabels()):
				item.set_fontsize(20)
			plt.text(0.03,0.9,'bc = '+str(round(bc,2))+'GHz, bw = '+str(round(bw/bc*100,1))+'%',fontsize=25,transform=ax.transAxes)
			#plt.show()
			figname = 'spec_row'+str(int(row))+'_col'+str(int(col))
			fn = os.path.join(out_fig, figname+'.png')
			pl.savefig(fn)
			
			bw_array[col][row] = bw/bc*100
			bc_array[col][row] = bc
			eff_array[col][row] = avereff*100
			detcol_array[col][row] = detcol
			detrow_array[col][row] = detrow
			detpol_array[col][row] = detpol
			interfg['r%dc%d'%(row,col)] = fb_hs_final
			spcos['r%dc%d'%(row,col)] = cs
			spsin['r%dc%d'%(row,col)] = ss		

	fnpickle = out_path_main + '%s_fts.pkl'%runn
	pickle.dump((bw_array, bc_array, eff_array, detcol_array, detrow_array, detpol_array, interfg, spcos, spsin, v_mirror, nSample),open(fnpickle,'w'))
	
	
if __name__=='__main__':
    main()
