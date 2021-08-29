import os, sys
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np


def plot_1Dhist(array1D, outpath, filename, ext='png', maintitle='', xlabel='', xunit='', ifylabel=True, nbins=15, binrange=None, histtype='step', facecolor='white', edgecolor='b', ifnotes = [True, True, True, True], stdtype='std') :
	pl.figure(figsize=(6,6), dpi=80)
	pl.clf()
	ax = pl.subplot(1,1,1)
	if binrange:
		n, bins, patches = plt.hist(array1D[~np.isnan(array1D)], 
					bins=nbins, 
					range=binrange, 
					histtype=histtype, 
					facecolor=facecolor, 
					edgecolor=edgecolor)
	else:
		 n, bins, patches = plt.hist(array1D[~np.isnan(array1D)], 
					bins=nbins, 
					histtype=histtype, 
					facecolor=facecolor, 
					edgecolor=edgecolor)
	if ifnotes[0]:
		valid=np.sum(np.array(n))
		pl.text(0.6, 0.95, 'valid = %d'%(valid), fontsize=15, transform=ax.transAxes)
	if ifnotes[1]:
		mean=np.nanmean(array1D)
		pl.text(0.6, 0.9, 'mean = %.2f'%(mean), fontsize=15, transform=ax.transAxes)
	if ifnotes[2]:
		median=np.nanmedian(array1D)
		pl.text(0.6, 0.85, 'median = %.2f'%(median), fontsize=15, transform=ax.transAxes)
	if ifnotes[3]:
		if stdtype=='std':
			if binrange:
				array1D_=array1D[np.where((array1D<binrange[1])&(array1D>binrange[0]))[0]]
				std=np.nanstd(array1D_)
			else:
				std=np.nanstd(array1D)
		elif stdtype=='mad':
			y=np.abs(array1D-np.nanmedian(array1D))
			std=np.nanmedian(y)*1.4826
		else:
			print('unknown stdtype %s'%stdtype)
		pl.text(0.6, 0.8, '%s = %.2f'%(stdtype,std), fontsize=15, transform=ax.transAxes)
	
	pl.suptitle(maintitle, fontsize=15)
	pl.xlabel('%s [%s]'%(xlabel, xunit), fontsize=15)
	if ifylabel:
		pl.ylabel('counts', fontsize=15)
	plt.tick_params(labelsize=14)
	if not os.path.isdir(outpath):
		os.makedirs(outpath)
	outfile = outpath + filename + '.' + ext
	pl.savefig(outfile, dpi=300)
	print(outfile)
	
	return valid, mean, median, std

def plot_1Dhist2(array1D, array1D2, outpath, filename, ext='png', maintitle='', xlabel='', xunit='', ifylabel=True, nbins=15, binrange=None, histtype='step', facecolor='white', edgecolor='b', edgecolor2='r', ifnotes = [True, True, True, True], stdtype='std') :
	pl.figure(figsize=(6,6), dpi=80)
	pl.clf()
	ax = pl.subplot(1,1,1)
	if binrange:
		n, bins, patches = plt.hist(array1D[~np.isnan(array1D)], 
					bins=nbins, 
					range=binrange, 
					histtype=histtype, 
					facecolor=facecolor, 
					edgecolor=edgecolor)
		n2, bins2, patches2 = plt.hist(array1D2[~np.isnan(array1D2)], 
					bins=nbins, 
					range=binrange, 
					histtype=histtype, 
					facecolor=facecolor, 
					edgecolor=edgecolor2)
	else:
		 n, bins, patches = plt.hist(array1D[~np.isnan(array1D)], 
					bins=nbins, 
					histtype=histtype, 
					facecolor=facecolor, 
					edgecolor=edgecolor)
		 n2, bins2, patches2 = plt.hist(array1D2[~np.isnan(array1D2)], 
					bins=nbins, 
					histtype=histtype, 
					facecolor=facecolor, 
					edgecolor=edgecolor2)
	if ifnotes[0]:
		valid=np.sum(np.array(n))
		pl.text(0.6, 0.95, 'valid = %d'%(valid), fontsize=15, transform=ax.transAxes)
	if ifnotes[1]:
		mean=np.nanmean(array1D)
		pl.text(0.6, 0.9, 'mean = %.2f'%(mean), fontsize=15, transform=ax.transAxes)
	if ifnotes[2]:
		median=np.nanmedian(array1D)
		pl.text(0.6, 0.85, 'median = %.2f'%(median), fontsize=15, transform=ax.transAxes)
	if ifnotes[3]:
		if stdtype=='std':
			if binrange:
				array1D_=array1D[np.where((array1D<binrange[1])&(array1D>binrange[0]))[0]]
				std=np.nanstd(array1D_)
			else:
				std=np.nanstd(array1D)
		elif stdtype=='mad':
			y=np.abs(array1D-np.nanmedian(array1D))
			std=np.nanmedian(y)*1.4826
		else:
			print('unknown stdtype %s'%stdtype)
		pl.text(0.6, 0.8, '%s = %.2f'%(stdtype,std), fontsize=15, transform=ax.transAxes)
	
	pl.suptitle(maintitle, fontsize=15)
	pl.xlabel('%s [%s]'%(xlabel, xunit), fontsize=15)
	if ifylabel:
		pl.ylabel('counts', fontsize=15)
	plt.tick_params(labelsize=14)
	if not os.path.isdir(outpath):
		os.makedirs(outpath)
	outfile = outpath + filename + '.' + ext
	pl.savefig(outfile, dpi=300)
	print(outfile)
	
	return valid, mean, median, std
