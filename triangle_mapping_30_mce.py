"""
Pseudocolor plots of unstructured triangular grids.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.io as sio
import numpy as np
import math
import sys,os
import ba30_ModuleMapping_BA as bamap
import ba30_ModuleMapping as skmap

Ndet_mce = 2*33
Ndet_det = 2*33
npixel = 16 # pixels in one tile
ntile = 1
lld = 4 # how many dets in one row pyhsically
w,h = 2*npixel,ntile

def prepare_triangles(lld):
	xy = []
	for i in range((lld+1)*(lld+1)):
		xy.append([])
		for j in range(2):
			xy[i].append(0)
	for i in range((lld+1)*(lld+1)):
		xy[i][0] = (i-i%(lld+1))/(lld+1)+1
		xy[i][1] = i%(lld+1)+1
	xy = np.asarray(xy)
	x = xy[:, 0]
	y = xy[:, 1]

	triangles = []
        for i in range((lld)*(2*lld)):
                triangles.append([])
                for j in range(3):
                        triangles[i].append(0)
	for i in range(lld):
		for j in range(lld*2):
			if j<lld:
				triangles[(j*lld+i)][0] = j*(lld+1)+i
				triangles[(j*lld+i)][1] = j*(lld+1)+i+1
				triangles[(j*lld+i)][2] = j*(lld+1)+i+lld+1
			else:
				triangles[(j*lld+i)][0] = (j-(lld-1))*(lld+1)+i
				triangles[(j*lld+i)][1] = (j-(lld-1))*(lld+1)+i+1
				triangles[(j*lld+i)][2] = (j-(lld-1))*(lld+1)+i-lld
	triangles = np.asarray(triangles)
	return x,y,triangles

def prepare_data(input_inmce, itile=0, mapping='BA'):

	data_arr = [float('nan') for x in range(w)]
	#da = [0]*h
	DATA = []
	IDet_col = []
	IDet_row = []
	POLAR = []
	for icol in [0,1]:
		for irow in range(33):
			DATA.append(input_inmce[icol][irow])
			if mapping=="BA":
				[im, dc,dr,dp] = bamap.mce2det(2*itile+icol,irow)
			else:
				[im, dc,dr,dp] = skmap.mce2det(icol,irow)
			IDet_col.append(dc)
			IDet_row.append(dr)
			POLAR.append(dp)
	for idet in range(Ndet_mce):
		if POLAR[idet] == 'A':
               
			if IDet_col[idet]<0 or IDet_col[idet]<0:
                                continue
                        idet_col = int(IDet_col[idet])
                        idet_row = int(IDet_row[idet])
			data_arr[(idet_col-1)*lld+(idet_row-1)] = DATA[idet]

                elif POLAR[idet] == 'B':
			
			if IDet_col[idet]<0 or IDet_col[idet]<0:
                                continue
                        idet_col = int(IDet_col[idet])
                        idet_row = int(IDet_row[idet])
                        data_arr[(idet_col-1)*lld+(idet_row-1)+npixel] = DATA[idet]
                else:
                        continue
	#for itile in range(ntile):
	#	da[itile] = np.asarray(data_arr[itile])

	return data_arr

def plot_a_map(data_array,triangles,x,y,mask_wire,itile,value_min,value_max,plot_title,plot_unit,out_path,maintitle=' ',ifshow=False, cmap='jet'):

	fig = plt.figure()
        plt.gca().set_aspect('equal')

        data_array = np.ma.masked_where(np.isinf(mask_wire),data_array)
        data_array = np.ma.masked_where(np.isnan(data_array),data_array)
        plt.tripcolor(x, y, triangles, facecolors=np.array([float(np.isinf(mm)+1)*0.5 for mm in mask_wire]), cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0 , edgecolors='k')
	plt.tripcolor(x, y, triangles, facecolors=data_array, cmap=cmap, vmin=value_min, vmax=value_max , edgecolors='k')
        cbar=plt.colorbar()
	cbar.set_label(plot_title+','+plot_unit, rotation=270, fontsize=15,labelpad=20)
        plt.title(maintitle,fontsize=15)
        plt.xlabel('column',fontsize=15)
        plt.ylabel('row',fontsize=15)
        plt.tick_params(axis='both',which='major',labelsize=15)
        plt.text(1.2, 1.2, 'A', fontsize=15)
        plt.text(1.6, 1.6, 'B', fontsize=15)
	plt.tight_layout()
	print(out_path)
        fig.savefig(out_path, dpi=300)
	if ifshow:
        	plt.show()

# vari = put name of variable beside the cbar
# maintitle = title of the plot, on the top
def triange_mapping_single(input_inmce, itile, vmin,vmax,vari,unit,outpath, mask_wire = None, mapping='BA', maintitle=' ', cmap = 'jet'):
	[x,y,triangles] = prepare_triangles(lld)
	da = prepare_data(input_inmce, mapping=mapping)
	if not mask_wire:
		mask_wire = get_default_mask()
        plot_a_map(da, 
		triangles, x, y, 
		mask_wire, itile, 
		vmin, vmax, vari, unit, 
		outpath, 
		maintitle=maintitle, 
		cmap = cmap)

def get_default_mask():
	mask_wire = [ 0,0,0,0,  #1A
                      0,0,0,0,     #2A
                      0,0,0,0,        #3A
                      0,0,0,0,     #4A
                      0,0,0,0,  #1B
                      0,0,0,0,        #2B
                      0,0,0,0,     #3B
                      0,0,0,0]     #4B
	return mask_wire
