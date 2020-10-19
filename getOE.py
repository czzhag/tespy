#!/usr/bin/env python

""" Update 10/24/2018
Very robust, just run:
python sk_light_analysis.py [fn300] [fn077]
in_path should be changed inside.
"""


import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import math
import sys, os
import os.path
import pylab as pl
import mce_data
#import calib_SK as calibf
#import lc_cooking_dark
#import single_pr_darkrun
from lcdata import get_LC as lc
from lcdata import get_PR as pr
import cPickle as pickle
#import ba40_ModuleMapping_BA as bamap
import shutil

# ================================== #
# INPUTs                             #
# ================================== #
class opts:
	testbed = 'SK'
	freq = 30 # GHz
	date = '20201016'
	module = 'BA30N5S5'
	runn = '%s_%s_%s'%(date,testbed,module)
	fn300 = 'LC_light_ba_FPU322mK_datamode1_source300k'
	fn077 = 'LC_light_ba_FPU315mK_datamode1_source077k_highbias_2'
	fitrange = {    'rnti_low': 3000.00,
	                'rnti_hgh': 6000.00,
	                'sc_low': None,
	                'sc_hgh': None}
	cols = [0,1]
	rows = range(33)
	ifoutputinMCE = False 
	doFigure = True
	# plot ax[1] xy limits
	pmin = 0.  # pW
	pmax = 100. # pW
	rmin = 50. # mOhm
	rmax = 300.# mOhm
	# for finding psat
	reasonableInorm = 30.0e-6 #A 

if opts.testbed in ['SK','sk']:
	import calib_SK
	calib = calib_SK.calib_sk()
	if opts.freq == 40:
		import ba40_ModuleMapping as minfo
	elif opts.freq == 30:
		import ba30_ModuleMapping as minfo
	else:
		print('***!!! wrong input')
		print(opts)
elif opts.testbed in ['BA','ba']:
        import calib_BA1
        calib = calib_BA1.calib_ba1()
        if opts.freq == 40:
        	import ba40_ModuleMapping_BA as minfo
        elif opts.freq == 30:
        	import ba30_ModuleMapping_BA as minfo
        else:
        	print('***!!! wrong input')
        	print(opts)
else:
	print('***!!! wrong input')
	print(opts)




def main():

	#===================================#
	# Loading data
	#===================================#
	in_path  = '../cryo/%s/'%opts.date
	out_path_main = '../output/%s/%s/'%(opts.date, opts.runn)

	
	out_path = out_path_main
	if not os.path.isdir(out_path):
		os.makedirs(out_path)
	shutil.copy2(os.path.realpath(__file__), out_path_main + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
	datafn300 = in_path + opts.fn300 + '/' + opts.fn300
	datafn077 = in_path + opts.fn077 + '/' + opts.fn077
	
	biasfn300 = datafn300 + '.bias'
	f300 = mce_data.MCEFile(datafn300)
	dname300 = os.path.split(datafn300)[0]
	biasfn077 = datafn077 + '.bias'
	f077 = mce_data.MCEFile(datafn077)
	dname077 = os.path.split(datafn077)[0]
	
	#===================================#
	# Calib
	#===================================#
	
	bias300 = np.loadtxt(biasfn300,skiprows=1)
	bias077 = np.loadtxt(biasfn077,skiprows=1)
	
	y300 = -1.0*f300.Read(row_col=True,unfilter='DC').data
	y077 = -1.0*f077.Read(row_col=True,unfilter='DC').data
	
	nr,nc,nt = y300.shape
	rows = np.zeros((nc,nr),dtype=np.int)
	cols = np.zeros((nc,nr),dtype=np.int)
	
	#===================================#
	# Outputs
	#===================================#
	dpdt_array = [[0 for i in range(33)] for j in range(24)]
	rnti_array = [[0 for i in range(33)] for j in range(24)]
	detcol_array = [[0 for i in range(33)] for j in range(24)]
	detrow_array = [[0 for i in range(33)] for j in range(24)]
	detpol_array = [[0 for i in range(33)] for j in range(24)]
	
	#===================================#
	# Cook LC
	#===================================#
	
	for row in opts.rows:
		for col in opts.cols:
			print('row %d, col %d'%(row, col))
			fitrange = opts.fitrange
			ibias300, ites300, rnti300, ksc300 = lc(bias300, y300[row,col], calib, fitrange, row, col)			
			ibias077, ites077, rnti077, ksc077 = lc(bias077, y077[row,col], calib, fitrange, row, col)
			rnti = (rnti300 + rnti077)/2.
			rnpsat = 1.3*rnti
			rr300, pp300, psat300 = pr(ibias300, ites300, calib, rnti, rnpsat, row=row, col=col, reasonableInorm=opts.reasonableInorm)
			rr077, pp077, psat077 = pr(ibias077, ites077, calib, rnti, rnpsat, row=row, col=col, reasonableInorm=opts.reasonableInorm)
			dpdt = (psat077 - psat300)/(300.-77.)*1.0e12
			print("dpdt %.3f pW/K"%dpdt)
			# Outputs
			if (rnti>0.25 or rnti<0.01) or (dpdt>0.12 or dpdt<0.001):
				dpdt = float('nan')
				rnti   = float('nan')
			dpdt_array[col][row] = dpdt # in pW/K
			rnti_array[col][row] = rnti*1000.0 # in mOhms
				
			im,detcol,detrow,detpol = minfo.mce2det(col,row)
			detcol_array[col][row] = detcol # in pW/K
			detrow_array[col][row] = detrow # in mOhms
			detpol_array[col][row] = detpol # in mOhms
			
			# plotting out i-v and p-r
			out_path_fig = out_path+'optfig/'
			if not os.path.isdir(out_path_fig):
				os.makedirs(out_path_fig)
			
			if opts.doFigure:
				pl.figure(figsize=(12,6), dpi=80)
				pl.clf()
				
				ax = pl.subplot(1,2,1)
				pl.plot(ibias300*1e6,ites300*1e6,'r',
					ibias077*1e6,ites077*1e6,'b')
				pl.text(0.05, 0.9, 'R = %.2f'%(rnti*1000)+r'$m\Omega$', fontsize=15, transform=ax.transAxes)     
				pl.xlabel('Ib [uA]', fontsize=15)
				pl.ylabel('Is [uA]', fontsize=15)
				pl.grid()
				
				ax = pl.subplot(1,2,2)
				pl.suptitle('Row %02d'%row + ' Col %02d'%col +' i-v and p-r', fontsize=15)
				pl.plot(rr300*1.0e3,pp300*1.0e12,'r', rr077*1.0e3,pp077*1.0e12,'b') # pW-mOhms
				plt.axvline(x=rnpsat*1.0e3,color='g',linestyle='--')
				plt.scatter([rnpsat*1.0e3],[psat077*1.0e12], s=100, color='b', marker='+')
				plt.scatter([rnpsat*1.0e3],[psat300*1.0e12], s=100, color='r', marker='+')
				
				pl.ylim(opts.pmin, opts.pmax)
				pl.xlim(opts.rmin, opts.rmax)
				
				pl.text(0.05, 0.9, 'dp/dt = %.3f pW/K'%dpdt, fontsize=15, transform=ax.transAxes)     
				pl.xlabel('R [mohms]', fontsize=15)
				pl.ylabel('p [pW]' , fontsize=15)
				pl.grid()
				
				#if row in range(33):
				if opts.ifoutputinMCE:
					fn = os.path.join(out_path_fig,'opteff_gcp%d_mcerow%02d'%(33*col+row+1, row) + '_mcecol%02d.png'%col)
				else:
					if detpol in ['A','B']:
						fn = os.path.join(out_path_fig,'opteff_im%d_detrow%02d'%(int(col/2), detrow) + '_detcol%02d'%detcol + '_pol'+ detpol+'.png')
					else:
						fn = os.path.join(out_path_fig,'opteff_mcerow%02d'%( row) + '_mcecol%02d.png'%col)
				pl.savefig(fn)
			

			
	fnpickle = out_path + opts.runn + '_dpdt_rnti.pkl'
	pickle.dump((dpdt_array, rnti_array, detcol_array, detrow_array, detpol_array),open(fnpickle,'w'))
	
	exit()
			
if __name__=='__main__':
    main()
            
