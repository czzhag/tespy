import matplotlib.pyplot as plt
import pylab as pl

def make2dscattering(fig, ax, x_, y_, s=80., color="k", marker="+", label="-"):
	x = x_.flatten()
	y = y_.flatten()
	ax.scatter(x, y, s, color=color, marker=marker, label=label)
	return fig, ax

def presetting(sx=6,sy=6,lx="x label",ly="ylabel"):
	fig = pl.figure(figsize=(sx,sy), dpi=80)
	pl.clf()
	ax = pl.subplot(1,1,1)
	labelsize = max([15, int(2*min([sx, sy]))])
	pl.xlabel(lx, fontsize=labelsize)
	pl.ylabel(ly, fontsize=labelsize)
	ticksize = labelsize
	pl.tick_params(labelsize=ticksize)
	return fig,ax      

def premultps(h = 1, w = 2, sx=7., sy=6.5, lx=["x label"], ly=["ylabel"], width_ratio = None):
        
	plt.clf()

	if lx == "x label":
		lxs = [[lx[0] for ii in range(w)] for jj in range(h)]
	elif h==1:
		if len(lx) == w:
			lxs = lx
		elif len(lx) == 1:
			if len(lx[0])==w:
				lxs = lx
			else:
				print("***Wrong x-label dimension.***")
				lxs = [["x label" for ii in range(w)] for jj in range(h)]
		else:
			print("***Wrong x-label dimension.***")
                        lxs = [["x label" for ii in range(w)] for jj in range(h)]
	else:
		if len(lx) == h:
			if len(lx[0])==w:
				lxs = lx
			elif w==1:
				lxs = lx
			else:
				print("***Wrong x-label dimension.***")
                                lxs = [["x label" for ii in range(w)] for jj in range(h)]
		else:
			print("***Wrong x-label dimension.***")
                        lxs = [["x label" for ii in range(w)] for jj in range(h)]

	if ly == "y label":
                lys = [[ly[0] for ii in range(w)] for jj in range(h)]
        elif h==1:
                if len(ly) == w:
                        lys = ly
                elif len(ly) == 1:
                        if len(ly[0])==w:
                                lys = ly
			else:
                                print("***Wrong y-label dimension.***")
                                lys = [["y label" for ii in range(w)] for jj in range(h)]
                else:
                        print("***Wrong y-label dimension.***")
                        lys = [["y label" for ii in range(w)] for jj in range(h)]
        else:
                if len(ly) == h:
                        if len(ly[0])==w:
                                lys = ly
			elif w==1:
				lys = ly
                        else:
                                print("***Wrong y-label dimension.***")
                                lys = [["y label" for ii in range(w)] for jj in range(h)]
                else:
                        print("***Wrong y-label dimension.***")
                        lys = [["y label" for ii in range(w)] for jj in range(h)]

	if width_ratio:
		fig, axs = plt.subplots(h, w, figsize=(sx*w, sy*h), gridspec_kw={'width_ratios': width_ratio})
	else:
		fig, axs = plt.subplots(h, w, figsize=(sx*w, sy*h))
	labelsize = int(2*min([sx*w, sy*h]))
	ticksize = labelsize 
	if h == 1 or w==1:
        	for ii,axi in enumerate(axs):
        		axi.set_xlabel(lxs[ii], fontsize=labelsize)
        		axi.set_ylabel(lys[ii], fontsize=labelsize)
        		axi.tick_params(labelsize=ticksize)
	else:
		for ih in range(h):
			for ii,axi in enumerate(axs[ih]):
                        	axi.set_xlabel(lxs[ih][ii], fontsize=labelsize)
                        	axi.set_ylabel(lys[ih][ii], fontsize=labelsize)
                        	axi.tick_params(labelsize=ticksize)
        return fig,axs

# settings of legend, grid, saving and showing
# fig: output of presetting func.
# ffn: full figure path+name.
# 
def possetting(fig, ffn = 'nan', ifleg = True, legloc = 1, legfontsize = 14, ifgrid = True, ifshow = False, legmatchtextcolor=True):
	if ifleg:
		leg = plt.legend(loc=legloc, prop={'size': legfontsize})
		if legmatchtextcolor:
			for line, text in zip(leg.get_lines(), leg.get_texts()):
				text.set_color(line.get_color())
	if ifgrid:
		plt.grid()
	if not ffn == 'nan':
		plt.tight_layout()
		fig.savefig(ffn)
	if ifshow:
		plt.show()
	plt.close()
	return

def posmultps(fig, axs, ffn = 'nan', ifleg = True, legloc = 1, legfontsize = 14, ifgrid = True, ifshow = True, legmatchtextcolor=True):
	
	if len(axs.shape) > 1:
		if isinstance(legloc, list):
			if not legloc.shape == axs.shape:
				print('***Given legloc doesnt match the dimension.***')
				legloc = [[1 for ii in range(axs.shape[1])] for jj in range(axs.shape[0])]
		else:
			legloc = [[legloc for ii in range(axs.shape[1])] for jj in range(axs.shape[0])]
		for ih in range(axs.shape[0]):
			for ii, axi in enumerate(axs[ih]):
				if ifleg:
                                       	axi.legend(loc=legloc[ih][ii], prop={'size': legfontsize})
                		if ifgrid:
                        		axi.grid()
	else:
		if isinstance(legloc, list):
			if not legloc.shape == axs.shape:
				print('***Given legloc doesnt match the dimension.***')
				legloc = [1 for ii in range(axs.shape[0])]
		else:
			legloc = [legloc for ii in range(axs.shape[0])]
		for ii, axi in enumerate(axs):
			if ifleg:
                        	leg = axi.legend(loc=legloc[ii], prop={'size': legfontsize})
				if legmatchtextcolor:
					for line, text in zip(leg.get_lines(), leg.get_texts()):
						text.set_color(line.get_color())
                        if ifgrid:
                                axi.grid()
	
	plt.tight_layout()

        if not ffn == 'nan':
                fig.savefig(ffn)
        if ifshow:
                plt.show()
        plt.close()
        return

