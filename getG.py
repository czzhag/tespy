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

from Gtemplists import gettemplists

# ================================== #
# INPUTs			     #
# ================================== #
class opts:
	testbed = 'SK'
	freq = 30 # GHz
	date = '20201205'
	module = 'BA30N5S5'
	runn = '%s_%s_%s'%(date,testbed,module)
	rnpsat = 0.07 # Ohm
	iffitbeta = True 
	fitrange = {	'rnti_low': 3000.00,
			'rnti_hgh': 5000.00,
			'sc_low': 0.00,
			'sc_hgh': 30.00}	
	#cols = range(12)
	cols = [0, 1]
	rows = range(33)


# ================================== #

templists = gettemplists()
templist = templists[opts.date]

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

	Gfn = '%s_G_%s_%s'%(opts.testbed,opts.module,opts.date)
	lcdata = {}
	prdata = {}
	colors = pl.cm.jet(np.linspace(0,1,len(templist)))
	
	in_path  = '../cryo/%s/'%(opts.date)
	print('input folder: %s'%in_path)
	out_path_main = '../output/%s/'%(opts.date)
	print('output folder: %s'%out_path_main)
	
	for temp in templist:
		#===================================#
		# loading data/making output dirs
		#===================================#
		filename = 'LC_G_FPU_'+str(temp)+'mK_datamode1_run1'
		if not os.path.isdir(out_path_main):
                        os.makedirs(out_path_main)
		out_path = out_path_main + filename + '/'
		if not os.path.isdir(out_path):
			os.makedirs(out_path)
		out_Gplot = out_path_main + '%s_Gplot/'%(opts.runn)
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
				ibias, ites, rnti, ksc = lc(bias, y[row,col], calib, fitrange, row, col)
				print(ksc)
				# get_PR_Ti(biascalib, fbcalib, calib, rnti, rnpsat, ksc = None, row = None, col = None, out_path = None, flip = 1)
				rr, pp, psat = pr(ibias, ites, calib, rnti, opts.rnpsat, ksc=ksc, row=row, col=col)
				lcdata[str(temp)+'_r'+str(row)+'c'+str(col)] = [ibias, ites]
				prdata[str(temp)+'_r'+str(row)+'c'+str(col)] = [rr, pp, rnti, psat, 1/(ites*rr)] 	
	
	#============================================#
	# Get psat and rn, do Gfit
	#============================================#

	templist.sort()
	#templist2 = [388, 397, 403, 411, 426, 439, 457, 474] # FPU1to2
	iteslist = [0]*len(templist)
	ibiaslist = [0]*len(templist)
	rnpsatlowlist = [0]*len(templist)
	rnpsathighlist = [0]*len(templist)
	psatlist = [0]*len(templist)
	rntilist = [0]*len(templist)
	GTdata = {}
	DetInfos = {}
	for col in opts.cols:
		for row in opts.rows:
			gtdata = {}
			dinfos = {}
			im, detcol,detrow,detpol = minfo.mce2det(col,row)
			for itt in range(len(templist)):
				psatlist[itt] = prdata[str(templist[itt])+'_r'+str(row)+'c'+str(col)][3]*1.00e12
                		rntilist[itt] = prdata[str(templist[itt])+'_r'+str(row)+'c'+str(col)][2]*1.00e3
				if rntilist[itt] < 20 or rntilist[itt]>1000:
					rntilist[itt] = float('nan')
			#rnti = np.nanmean(rntilist)
			rntilist = np.array(rntilist)	
			rnti = np.mean(rntilist[~np.isnan(rntilist)])
			xdata = np.linspace(240, 510, 100)

			fig = pl.figure(figsize=(20,6.5), dpi=80)
			pl.clf()

			ax = pl.subplot(1, 3, 1)
                        pl.suptitle('Row %02d'%row + ' Col %02d'%col)
                        pl.xlabel('Ibias [uA]', fontsize=15)
                        pl.ylabel('Ites [uA]', fontsize=15)
                        pl.xlim(0,350)
                        pl.ylim(0,50)
                        for itemp,temp in enumerate(templist):
				#temp2 = templist2[itemp] #FPU1to2
                                pl.plot(lcdata[str(temp)+'_r'+str(row)+'c'+str(col)][0]*1.e6,
					lcdata[str(temp)+'_r'+str(row)+'c'+str(col)][1]*1.e6, 
					color=colors[itemp], 
					label=str(temp)+'mK') #FPU1to2
			plt.tick_params(labelsize=14)
			pl.grid()
                        pl.legend(loc=1, prop={'size': 14})

			ax = pl.subplot(1, 3, 2)
	                pl.suptitle('Row %02d'%row + ' Col %02d'%col)
        	        pl.xlabel('R [mOhms]', fontsize=15)
                	pl.ylabel('P [pW]', fontsize=15)
			pl.xlim(0,200)
			pl.ylim(0,6)
			for itemp,temp in enumerate(templist):
				#temp2 = templist2[itemp] #FPU1to2
				pl.plot(prdata[str(temp)+'_r'+str(row)+'c'+str(col)][0]*1.00e3,
					prdata[str(temp)+'_r'+str(row)+'c'+str(col)][1]*1.00e12,
					color=colors[itemp],
					label=str(temp)+'mK') #FPU1to2
			plt.tick_params(labelsize=14)
			pl.grid()
			pl.legend(loc=1, prop={'size': 14})
			pl.axvline(x = opts.rnpsat*1.0e3,color='r', linestyle='--')

			ax = pl.subplot(1, 3, 3)
			pl.scatter(templist,psatlist) #FPU1to2
			if 'popt' in locals():
				del popt
				del pcov
			
			try:
				ind = np.where(np.array(psatlist) > 0.1)
                        	popt, pcov = curve_fit(GTcModel, 
						np.array(templist)[ind], 
						np.array(psatlist)[ind]) #FPU1to2
				if opts.iffitbeta:
					pl.plot(xdata, GTcModel(xdata, *popt), 'g--',label='fit: Gc=%5.3f, Tc=%5.3f, beta=%5.3f' % tuple(popt))
				else:
					pl.plot(xdata, GTcModel(xdata, *popt), 'g--',label='fit: Gc=%5.3f, Tc=%5.3f, beta=2.00' % tuple(popt))
			except:
				popt = np.array([-1,-1,-1])
				pcov = np.array([-1,-1,-1])
				pass
			
			pl.xlim(200,510)
                        pl.ylim(0,6)
			pl.ylabel('P [pW]', fontsize=15)
			pl.xlabel('T [mK]', fontsize=15)
			plt.tick_params(labelsize=14)
			plt.grid()
			if popt[0]>0 and popt[1]>0 and popt[1]<550:
				try:
					pl.text(0.5,0.9,'Gc='+str(round(popt[0]*1e3,1))+' pW/K',transform=ax.transAxes, fontsize=14)
					pl.text(0.5,0.85,'Tc='+str(round(popt[1]))+' mK',transform=ax.transAxes, fontsize=14)
					if opts.iffitbeta:
						pl.text(0.5,0.80,'beta='+str(round(popt[2],2)),transform=ax.transAxes, fontsize=14)
					else:
						pl.text(0.5,0.80,'beta='+str(round(2.0,2)),transform=ax.transAxes, fontsize=14)
					pl.text(0.5,0.75,'RnTi='+str(round(rnti))+' mOhm',transform=ax.transAxes, fontsize=14)
					gtdata['Gc'] = popt[0]*1e3
	        		        gtdata['Tc'] = popt[1]
					if opts.iffitbeta:
        			        	gtdata['beta'] = popt[2]
					else:
						gtdata['beta'] = 2.00
					gtdata['rnti'] = rnti	
					dinfos['mcecol'] = col
					dinfos['mcerow'] = row
					dinfos['detcol'] = detcol
                		        dinfos['detrow'] = detrow
                		        dinfos['pol'] = detpol
				except:
                		        gtdata['Gc'] = float('nan') #pW/K
                		        gtdata['Tc'] = float('nan')
                		        gtdata['beta'] = float('nan')
					gtdata['rnti'] = float('nan') # mO
					dinfos['mcecol'] = float('nan')
                		        dinfos['mcerow'] = float('nan')
                		        dinfos['detcol'] = float('nan')
                		        dinfos['detrow'] = float('nan')
                		        dinfos['pol'] = float('nan')
			else:
				gtdata['Gc'] = float('nan')
                                gtdata['Tc'] = float('nan')
                                gtdata['beta'] = float('nan')
                                gtdata['rnti'] = float('nan')
                                dinfos['mcecol'] = float('nan')
                                dinfos['mcerow'] = float('nan')
                                dinfos['detcol'] = float('nan')
                                dinfos['detrow'] = float('nan')
                                dinfos['pol'] = float('nan')

				
			GTdata['r'+str(row)+'c'+str(col)] = gtdata	
                        DetInfos['r'+str(row)+'c'+str(col)] = dinfos
			del gtdata
			del dinfos	
			
			fn = os.path.join(out_Gplot,'%s_G_row%d'%(opts.testbed, row) + '_col%d'%col)
			pl.suptitle('mcerow %02d mcecol %02d, detRow %02d detCol %02d Pol-%s'%(row,col,detrow,detcol,detpol), fontsize=15)
			fnd = os.path.join(out_Gplot,'%s_G_detrow%d'%(opts.testbed, detrow) + '_detcol%d'%detcol+'_pol%s'%detpol)
			pl.savefig(fn)	
			pl.savefig(fnd)
			 
	shutil.copy2(os.path.realpath(__file__), out_Gplot + (os.path.realpath(__file__).split("/")[-1]).replace(".py",".txt"))
	fnpickle = out_Gplot + Gfn + '.pkl'
        pickle.dump((lcdata,prdata,GTdata,DetInfos),open(fnpickle,'w'))
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
            
