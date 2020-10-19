import numpy as np
import mce_data
import sys,os
sys.path.insert(0, "/home/cheng/analysis/ploting")
import easyplots as eps
import matplotlib.pyplot as plt
import pylab as pl

def get_Ic(ites, vtes):
	dv    = vtes[1:] - vtes[:-1]
	di    = ites[1:] - ites[:-1]
	if len(np.where((dv<0.00e-6))[0])>0:
		vtes = vtes[np.where((dv<0.00e-6))[0][-1]+1:]
		ites = ites[np.where((dv<0.00e-6))[0][-1]+1:]
	Ic = np.interp(min(vtes)+0.001e-6, vtes, ites)
		#np.where((dv<0.00e-6))[0][-1]
		#Ic    = ites[np.where((dv>0.00e-6))[0][0]]
	#else:
		#Ic    = float('nan')
		#Ic = np.interp(min(vtes), vtes, ites)
		#Ic = ites[0]
	return Ic

# for LC response
# input ites should be in physical unit,
# in bias increasing direction
def fixfluxramp(ites):
	ites0 = ites[:-1]
	ites1 = ites[1:]
	di    = ites1 - ites0
	jumps = np.where(abs(di)>400e-6)
	for ijump in jumps[0]:
		ites[ijump+1:] += -di[ijump] + di[ijump-1]
	return ites

# input raw bias/fb in ADU, only for 1 det
# output biascalib, fbcalib with shift, rnti
def get_LC_calibed(bias, fb, calib, fitrange = None, row = None, col = None, out_path = None, flip = 1, DCflag='RN'):

        bias = bias/calib["BIAS_CAL"][0];
        fb   = fb/calib["FB_CAL"][0];

	if not fitrange:
		fitrange = get_default_fitrange()

	fbcalib_ = -1*flip*fb*calib["FB_CAL"][0]
	biascalib_ = bias*calib["BIAS_CAL"][0]
	
        # Fit
        # fit range
        bias_fit = np.array([bias[i] for i in xrange((len(bias))) if (bias[i]>fitrange["rnti_low"] and bias[i]<fitrange["rnti_hgh"])])
        fb_fit   = np.array([fb[i] for i in xrange((len(fb))) if (bias[i]>fitrange["rnti_low"] and bias[i]<fitrange["rnti_hgh"])])
        # fit with poly1
        ceff  = np.polyfit(bias_fit*calib["BIAS_CAL"][0],-1*flip*fb_fit*calib["FB_CAL"][0],1)
        ffunc = np.poly1d(ceff)
        # normal resistance
        RR = calib["R_SH"]*(1.00/ceff[0]-1.00) #Ohms
        if RR<10e-3 or RR>1000e-3:
                RR = float('nan')
	# fit sc
	if fitrange["sc_low"] is None:
        	fitrange["sc_low"] = min(bias)
		fitrange["sc_hgh"] = max(bias)
	bias_sc  = np.array([bias[i] for i in xrange((len(bias))) if (bias[i]>fitrange["sc_low"] and bias[i]<fitrange["sc_hgh"])])
        fb_sc    = np.array([fb[i] for i in xrange((len(fb))) if (bias[i]>fitrange["sc_low"] and bias[i]<fitrange["sc_hgh"])])
        # fit sc with polfb1
        ceff_sc  = np.polyfit(bias_sc*calib["BIAS_CAL"][0],-1*flip*fb_sc*calib["FB_CAL"][0],1)
        ffunc_sc = np.poly1d(ceff_sc)

	if DCflag == 'RN':
		shift_h  = ffunc(0)
	elif DCflag == 'SC':
		shift_h  = ffunc_sc(0)
	elif DCflag == 0:
		shift_h  = fbcalib_[0]
	elif DCflag == -1:
		shift_h  = fbcalib_[-1]
	else:
		print("DCflag = ['RN', 'SC', 0, -1], otherwise use RN")
		shift_h  = ffunc(0)
        fbcalib = fbcalib_ - shift_h
	biascalib = biascalib_
        rntifit  = ffunc(biascalib) - shift_h

	if out_path and row+1 and col+1:
                fig,ax = eps.presetting(7.4,6,lx="Ib [uA]",ly="Ites [uA]")
                pl.suptitle('Row %02d'%row + ' Col %02d'%col)
                pl.plot(biascalib*1e6,fbcalib*1e6, biascalib*1e6, rntifit*1e6)
                pl.xlim(min(biascalib)*1e6,max(biascalib)*1e6)
                
                pl.text(0.6, 0.85, 'R = %.2f Ohms'%RR, fontsize=15, transform=ax.transAxes)
                pl.text(0.6, 0.75, 'SC slope = %.2f'%ceff_sc[0], fontsize=15, transform=ax.transAxes)
		
                fn = os.path.join(out_path,'single_iv_row%02d'%row + '_col%02d_yes.png'%col)
                eps.possetting(fig, ffn = fn, ifleg = False, ifgrid = True, ifshow = False)
        
	return biascalib, fbcalib, RR, ceff_sc[0]



# input raw bias/fb in ADU, only for 1 det
# output biascalib, fbcalib with shift, rnti
def get_LC(bias, fb, calib, fitrange = None, row = None, col = None, out_path = None, flip = 1, DCflag='RN'):
	
	if not fitrange:
		fitrange = get_default_fitrange()

	fbcalib_ = -1*flip*fb*calib["FB_CAL"][0]
	biascalib_ = bias*calib["BIAS_CAL"][0]
	
        # Fit
        # fit range
        bias_fit = np.array([bias[i] for i in xrange((len(bias))) if (bias[i]>fitrange["rnti_low"] and bias[i]<fitrange["rnti_hgh"])])
        fb_fit   = np.array([fb[i] for i in xrange((len(fb))) if (bias[i]>fitrange["rnti_low"] and bias[i]<fitrange["rnti_hgh"])])
        # fit with poly1
        ceff  = np.polyfit(bias_fit*calib["BIAS_CAL"][0],-1*flip*fb_fit*calib["FB_CAL"][0],1)
        ffunc = np.poly1d(ceff)
        # normal resistance
        RR = calib["R_SH"]*(1.00/ceff[0]-1.00) #Ohms
        if RR<10e-3 or RR>1000e-3:
                RR = float('nan')
	# fit sc
	if fitrange["sc_low"] is None:
        	fitrange["sc_low"] = min(bias)
		fitrange["sc_hgh"] = max(bias)
	bias_sc  = np.array([bias[i] for i in xrange((len(bias))) if (bias[i]>fitrange["sc_low"] and bias[i]<fitrange["sc_hgh"])])
        fb_sc    = np.array([fb[i] for i in xrange((len(fb))) if (bias[i]>fitrange["sc_low"] and bias[i]<fitrange["sc_hgh"])])
        # fit sc with polfb1
        ceff_sc  = np.polyfit(bias_sc*calib["BIAS_CAL"][0],-1*flip*fb_sc*calib["FB_CAL"][0],1)
        ffunc_sc = np.poly1d(ceff_sc)

	if DCflag == 'RN':
		shift_h  = ffunc(0)
	elif DCflag == 'SC':
		shift_h  = ffunc_sc(0)
	elif DCflag == 0:
		shift_h  = fbcalib_[0]
	elif DCflag == -1:
		shift_h  = fbcalib_[-1]
	else:
		print("DCflag = ['RN', 'SC', 0, -1], otherwise use RN")
		shift_h  = ffunc(0)
        fbcalib = fbcalib_ - shift_h
	biascalib = biascalib_
        rntifit  = ffunc(biascalib) - shift_h

	if out_path and row+1 and col+1:
                fig,ax = eps.presetting(7.4,6,lx="Ib [uA]",ly="Ites [uA]")
                pl.suptitle('Row %02d'%row + ' Col %02d'%col)
                pl.plot(biascalib*1e6,fbcalib*1e6, biascalib*1e6, rntifit*1e6)
                pl.xlim(min(biascalib)*1e6,max(biascalib)*1e6)
                
                pl.text(0.6, 0.85, 'R = %.2f Ohms'%RR, fontsize=15, transform=ax.transAxes)
                pl.text(0.6, 0.75, 'SC slope = %.2f'%ceff_sc[0], fontsize=15, transform=ax.transAxes)
		
                fn = os.path.join(out_path,'single_iv_row%02d'%row + '_col%02d_yes.png'%col)
                eps.possetting(fig, ffn = fn, ifleg = False, ifgrid = True, ifshow = False)
        
	return biascalib, fbcalib, RR, ceff_sc[0]


#ksc : superconducting slope from the lc func above, not required.
def get_PR_Ti(biascalib, fbcalib, calib, rnti, rnpsat, ksc = None, row = None, col = None, out_path = None, flip = 1):
	
	if not (out_path and row+1 and col+1):
		doFigure = False
	else:
		doFigure = True

	rr = (biascalib/fbcalib-1)*calib["R_SH"]
	pp = fbcalib*fbcalib*rr
	if ksc:
		Inorm = np.where((fbcalib<(ksc + 0.01)*biascalib*calib["R_SH"]/(calib["R_SH"]+rnti)) 
			& (fbcalib>(ksc - 0.01)*biascalib*calib["R_SH"]/(calib["R_SH"]+rnti)), 
			fbcalib, float('nan'))
	else:
		Inorm = np.where((fbcalib<1.1*biascalib*calib["R_SH"]/(calib["R_SH"]+rnti))
                        & (fbcalib>0.9*biascalib*calib["R_SH"]/(calib["R_SH"]+rnti)),
                        fbcalib, float('nan'))
	
	while 1:
		minInorm = np.nanmin(Inorm)
		ind = np.where((pp<=rnti*(minInorm*1.2)**2)&(pp>rnti*(0.8*minInorm)**2))[0]
		if len(ind) or (len(Inorm[~np.isnan(Inorm)]) == 0):
			break
		else:
			Inorm[np.nanargmin(Inorm)]=float('nan')
	if len(ind):
		psat = np.interp(rnpsat, rr[ind][::-1], pp[ind][::-1])
		if psat < 0 or psat > 1000e-12:
			psat = float('nan')
	else:
		psat = float('nan')
	
	#===================================#
	# Plot
	#===================================#
	if doFigure:
		pl.clf()
		pl.suptitle('Row %02d'%row + ' Col %02d'%col)
		pl.xlabel('R [mOhms]', fontsize=15)
		pl.ylabel('P [pW]', fontsize=15)
		pl.plot(rr*1.00e3,pp*1.00e12, color='k', linewidth=2)
		if (not np.isnan(rnti)) and (not np.isinf(rnti)) and rnti>0:
			pl.xlim(0,rnti*1e3*1.2)
			pl.ylim(0,10)
		fn = os.path.join(out_path,'single_pr_row%02d'%row + '_col%02d_yes.png'%col)
		pl.axvline(x=rnti*1.0e3, color='r', linestyle='--', alpha=0.5)
		pl.axhline(y=psat*1.0e12, color='r', linestyle='--', alpha=0.5)
		pl.grid()
		pl.tight_layout()
		pl.savefig(fn)

	return rr, pp, psat



def get_PR(biascalib, fbcalib, calib, rnti, rnpsat, row = None, col = None, out_path = None, flip = 1, reasonableInorm = 30.0e-6):
	
	if not (out_path and row+1 and col+1):
		doFigure = False
	else:
		doFigure = True

	rr = (biascalib/fbcalib-1)*calib["R_SH"]
	pp = fbcalib*fbcalib*rr
	Inorm = np.where((fbcalib<1.1*biascalib*calib["R_SH"]/(calib["R_SH"]+rnti)) 
			& (fbcalib>0.9*biascalib*calib["R_SH"]/(calib["R_SH"]+rnti)), 
			fbcalib, float('nan'))
	while 1:
		maxInorm = np.nanmax(Inorm)
		ind = np.where((pp>rnti*(0.5*maxInorm)**2)&(pp<rnti*(1.2*maxInorm)**2))[0]
		if not( len(ind) or (len(Inorm[~np.isnan(Inorm)]) == 0)):
			Inorm[np.nanargmax(Inorm)]=float('nan')
		elif maxInorm > reasonableInorm:
			Inorm[np.nanargmax(Inorm)]=float('nan')
		else:
			break
	if len(ind):
		psat = np.interp(rnpsat, rr[ind][::-1], pp[ind][::-1])
		if psat < 0 or psat > 1000e-12:
			psat = float('nan')
	else:
		psat = float('nan')
	
	#===================================#
	# Plot
	#===================================#
	if doFigure:
		pl.clf()
		pl.suptitle('Row %02d'%row + ' Col %02d'%col)
		pl.xlabel('R [mOhms]', fontsize=15)
		pl.ylabel('P [pW]', fontsize=15)
		pl.plot(rr*1.00e3,pp*1.00e12, color='k', linewidth=2)
		if (not np.isnan(rnti)) and (not np.isinf(rnti)) and rnti>0:
			pl.xlim(0,rnti*1e3*1.2)
			pl.ylim(0,10)
		fn = os.path.join(out_path,'single_pr_row%02d'%row + '_col%02d_yes.png'%col)
		pl.axvline(x=rnti*1.0e3, color='r', linestyle='--', alpha=0.5)
		pl.axhline(y=psat*1.0e12, color='r', linestyle='--', alpha=0.5)
		pl.grid()
		pl.tight_layout()
		pl.savefig(fn)

	return rr, pp, psat



def get_SVs(ites, ibias):
	vv = (ibias-ites)*0.003
	ss = 1/vv	
	return ss,vv

# return a default fitrange
# based on BA1 setup
# in ADU
def get_default_fitrange():
	fitrange = {}
	fitrange["rnal_low"] = 9000
        fitrange["rnal_hgh"] = 10000
	fitrange["rnti_low"] = 1900
        fitrange["rnti_hgh"] = 2000
        fitrange["sc_low"] = 0
        fitrange["sc_hgh"] = 50
	return fitrange
	
