import os,sys
import mce_data
import calib_BA1
import calib_SK
import numpy as np

def get_raw_timestream(datafn, calfn='calib_BA1', count=None, start=0):

	f = mce_data.MCEFile(datafn)
	dname = os.path.split(datafn)[0]
	if calfn=='calib_BA1':
		calib = calib_BA1.calib_ba1()
	elif calfn=='calib_SK':
		calib = calib_SK.calib_sk()
	else:
		print('***wrong calib fn')
		return False
	
	#===================================#
	# Calib
	#===================================#
	# calib.FB_CAL = (calib.V_FB_MAX/(2^calib.BITS_FB))./(calib.R_FB+calib.R_WIRE) ./ calib.M_FB;
	y = -1.0*f.Read(row_col=True,unfilter='DC', count=count, start=start).data
	yCalib =  1.0*f.Read(row_col=True,unfilter='DC', count=count, start=start).data*calib['FB_CAL'][1]
	nr,nc,nt = y.shape
	
	class timestreams:
		info = vars(f)
		fb = y
		fbCalib = yCalib
	
	return timestreams

def remove_baseline(y, n=0, w_glitch=11, h_glitch=10):
	xp = np.linspace(0, len(y)-1, len(y))
	mask = np.zeros(len(y))
	lMean = np.correlate(y, np.ones(w_glitch), 'same') / np.correlate(np.ones(len(y)), np.ones(w_glitch), 'same')
	lVar = (np.correlate(y ** 2, np.ones(w_glitch), 'same') / np.correlate(np.ones(len(y)), np.ones(w_glitch), 'same') - lMean ** 2)
	noise = h_glitch*np.mean(lVar[int(w_glitch)+1:-int(w_glitch)-1])
	for ii,mm in enumerate(mask):
		if lVar[ii]>noise:
			if ii-2<0:
				mask[:ii+3] = 1
			elif ii+3>=len(y):
				mask[ii-2:] = 1
			else:
				mask[ii-2:ii+3] = 1
	idx = np.where(mask == 0)
	pbl = np.poly1d(np.polyfit(xp[idx], y[idx], n))
	bl = pbl(xp)
	yb = y - bl
	bl = bl
	yb = yb
	mask = mask
	return bl,yb,mask

