#!/usr/bin/env python

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import math
import cPickle as pickle
from scipy.optimize import curve_fit
import sys, os
import os.path
import shutil
import pylab as pl
import mce_data

from lcdata import get_LC as lc
from lcdata import get_PR_Ti as pr


# ================================== #
# INPUTs			     #
# ================================== #
class opts:
	testbed = 'BA'
	freq = 3040 # GHz
	date = '20200122'
	module = 'BA1'
	runn = '%s_%s_%s'%(date,testbed,module)
	rnpsat = 0.07 # Ohm
	iffitbeta = True 
	fitrange = {	'rnti_low': 850.00,
			'rnti_hgh': 1000.00,
			'sc_low': 0.00,
			'sc_hgh': 30.00}	
	cols = range(24)
	#cols = [0, 1]
	rows = range(33)


# ================================== #


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
		import ba40_ModuleMapping_BA as minfo
	else:
		print('***!!! wrong input')
		print(opts)
else:
	print('***!!! wrong input')
	print(opts)


if opts.iffitbeta:
	def GTcModel(T,G,Tc,beta):
		return G/(beta+1)*(Tc**(beta+1)-T**(beta+1))/Tc**beta
else:	
	def GTcModel(T,G,Tc):
		return G/(2+1)*(Tc**(2+1)-T**(2+1))/Tc**2

def main():

	Pfn = '%s_Psat_%s_%s'%(opts.testbed,opts.module,opts.date)
	lcdata = {}
	prdata = {}
	
	in_path  = './cryo/%s/'%(opts.date)
	print('input folder: %s'%in_path)
	out_path_main = './output/%s/'%(opts.date)
	print('output folder: %s'%out_path_main)
	
	#===================================#
	# loading data/making output dirs
	#===================================#
	filename = 'LC_dark_FPU_300mK_datamode1_runskyclear2'
	if not os.path.isdir(out_path_main):
                os.makedirs(out_path_main)
	out_path = out_path_main + filename + '/'
	if not os.path.isdir(out_path):
		os.makedirs(out_path)
	out_Gplot = out_path_main + '%s_plot/'%(opts.runn)
	if not os.path.isdir(out_Gplot):
                os.makedirs(out_Gplot)
	
	datafn = in_path + filename + '/' + filename
	
	biasfn = datafn + '.bias'
	f = mce_data.MCEFile(datafn)
	dname = os.path.split(datafn)[0]
			
	bias = np.loadtxt(biasfn,skiprows=1)
	y = -1.0*f.Read(row_col=True,unfilter='DC').data
	
	nr,nc,nt = y.shape
	rows = np.zeros((nc,nr),dtype=np.int)
	cols = np.zeros((nc,nr),dtype=np.int)
		
	
	#===================================#
	# Cook LC
	#===================================#
	for col in opts.cols:
		for row in opts.rows:
			fitrange = opts.fitrange
			# get_LC(bias, fb, calib, fitrange = None, row = None, col = None, out_path = None, flip = 1, DCflag='RN')
			ibias, ites, rnti, ksc = lc(bias, y[row,col], calib, fitrange, row, col, out_path=out_path)
			# get_PR_Ti(biascalib, fbcalib, calib, rnti, rnpsat, ksc = None, row = None, col = None, out_path = None, flip = 1)
			rr, pp, psat = pr(ibias, ites, calib, rnti, opts.rnpsat, row=row, col=col, out_path=out_path)
			lcdata['r'+str(row)+'c'+str(col)] = [ibias, ites]
			prdata['r'+str(row)+'c'+str(col)] = [rr, pp, rnti, psat, 1/(ites*rr)] 	
	
		
	shutil.copy2(os.path.realpath(__file__), out_Gplot + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
        Gfn = opts.runn
        fnpickle = out_Gplot + Gfn + '.pkl'
        pickle.dump((lcdata,prdata),open(fnpickle,'w'))
	''' 
        read data: d = pickle.l/ad(/pen(fnpickle,'r')) 
                d[0] --> lcdata 
                d[1] --> prdata 
                d[0]['457_r4c0']['rnti'/Ohms,'calibFB'/A,'calibBIAS'/A,'rntifit','sc_slope'] 
                d[1]['457_r4c0']['resistance'/Ohms,'power'/W,'psat'/W,'rnti'/Ohms, responsivity/(1/V)] 
		d[2] --> GTdata 'r28c1': {'rnti': 85.20458899931745, 'beta': 1.5068964400413496, 'Gc': 15.159194238818376, 'Tc': 482.51052453878344}
		d[3] --> DetInfos 'r20c1': {'pol': 'A', 'mcecol': 1, 'mcerow': 20, 'detcol': 4, 'detrow': 2}

        ''' 
       

			
if __name__=='__main__':
    main()
            
