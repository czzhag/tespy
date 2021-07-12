#!/usr/bin/env python

import sys, os
tespath = "/home/cheng/analysis/tespy"
sys.path.insert(0, tespath)
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from scipy import signal
import cPickle as pickle
import math
import mce_data
import peakdetect
filelistpath = "./filelist"
sys.path.insert(0, filelistpath)

class opts:
	if len(sys.argv)<2:
		date = None
	else:
		date = sys.argv[1]

	if len(sys.argv)<3:
		dn = None
	else:
		dn = sys.argv[2]

	if len(sys.argv)<4:
		mc = None
	else:
		mc = int(sys.argv[3])

	if len(sys.argv)<5:
		mr = None
	else:
		mr = int(sys.argv[4])

def get_time_stream(row, col, S, filename, out_path_main):
	
	out_path = out_path_main + 'fts_time_stream/'
	if not os.path.isdir(out_path):
		os.makedirs(out_path)
	
	pl.figure(figsize=(10,10), dpi=80)
	#y = signal.detrend(S[row,col])
	y = S[row,col]
	pl.plot(y)
	[maxp,minp] = peakdetect.peakdetect(y_axis=y, lookahead=5000)
	minx,miny = zip(*minp)
	minx,miny = real_wlf(minx,miny,lsurr=15000)
	pl.scatter(minx,miny,color='r')
	#plt.ylim(-200,200)
	fn = os.path.join(out_path, 'row'+str(int(row))+'col'+str(int(col))+'.png')
	print("'r%dc%d':{'"%(row,col)+filename+"':%s}"%str(minx))
	plt.show()
	plt.close()
	return S[row,col], minx,miny
	

def real_wlf(minx, miny, lsurr):
	xwlf = []
	ywlf = []
	for ii,x in enumerate(minx):
		ysurr = [miny[jj] for jj in range(len(miny)) if minx[jj]<x+lsurr and minx[jj]>x-lsurr]
		if miny[ii] == np.min(ysurr):
			xwlf.append(x)
			ywlf.append(miny[ii])
	return xwlf,ywlf

def find_wlf(in_path,filename,mc,mr):

	out_path = in_path.replace('cryo', 'output') 

	filepath = in_path+'/'+filename
	print(filepath)
	f = mce_data.MCEFile(filepath)
	dname = os.path.split(filepath)[0]
	S = f.Read(row_col=True,unfilter='DC').data
	y,xwlf,ywlf = get_time_stream(mr, mc, S, filename, out_path)
				

def main():
	if opts.date is None:
		opts.date = raw_input('date YYYYMMDD: ')
	if opts.dn is None:
		opts.dn = raw_input('filename: ')
	if opts.mc is None:
		opts.mc = int(raw_input('col/int: '))
	if opts.mr is None:
		opts.mr = int(raw_input('row/int: '))
	in_path = '/home/data/cryo/%s'%(opts.date)
	find_wlf(in_path,opts.dn,opts.mc,opts.mr)

if __name__=='__main__':
    main()
