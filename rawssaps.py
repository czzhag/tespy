import sys, os
import numpy as np
import scipy.io as sio
from scipy import signal
import easyplots as esp
import calib_BA1

calib = calib_BA1.calib_ba1()

#row = int(sys.argv[1])
#col = int(sys.argv[2])
for row in [28]:
    for col in [0,1,4,5,10]:
        d=sio.loadmat('/home/czhang/cryo/20200123/freeze_row%d_col%d.mat'%(row, col))
        y=d['trace']
        fs=50.e6
        nfft=round(len(y[0])/20.)
        ff, ps = signal.welch(y[0], fs, return_onesided=True, detrend='constant', nperseg=nfft)
        
        fig,ax=esp.presetting(6,5,'freq',' pA/rtHz')
        ax.plot(ff,(ps)**0.5*abs(calib['FB_CAL'][0])/350.*10*1e12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('row%d col%d'%(row, col))
        #ax.set_ylim(1e-11, 1e-2)
        #ax.grid()
        esp.possetting(fig, ffn='/home/czhang/output/20200124/freeze_row%d_col%d_ps_nfft%d.png'%(row, col, nfft),ifgrid=True, ifleg = False, ifshow=True)
