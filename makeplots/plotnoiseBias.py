import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt

modules=['N2','M4','N3','M5','M6','N1','M8','Mx2','M3','M7','N4','M9']
colors =['orange','yellow','limegreen','green','cyan','skyblue','blue','grey','b','salmon','pink','grey']

#
#workchns=[
#[3,4,6,15,16,17,18,19,25,26,27,29],#0
#[1,2,3,19,25],#1
#[7,9,10,15,16,17,18,20,25,27,28,29,30,31],#2
#[2,7,9,10,13,15,18,23,24,25,26,28,29,30],#3
#[3,4,5,13,16,17,18,24,27,28,29],#4
#[1,2,3,4,5,13,20,24,25,26,27,29],#5
#[15,17,18,19,21,25,26,27,28,29,30,32],#6
#[4,23,24,25,26,27,28],#7
#[3,4,5,6,7,9,10,15,16,17,18,19,24,25,26,27,28,29,30],#8
#[1,2,4,7,9,10,15,16,17,18,19,24,25,27,31],#9
#[3,18,19,23,24,26,27,28,30],#10
#[1,3,4,5,6,16,17,18,19,20,21,27,30,31,32],#11
#[3,4,5,6,7,9,10,13,15,17,18,19,21,23,24,25,26,27,28,29,30,31,32],#12
#[1,2,3,5,6,7,9,10,13,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32],#13
#[5,9,10,13,17,19,20,21,23,24,25,26,27,28,29,30,31,32],#14
#[1,2,3,4,6,7,9,13,15,16,17,18,20,23,24,25,26,27,30,31,32],#15
#[3,4,7,10,13,15,16,17,18,19,20,21,25,26,27,28,29,30,31,32],#16
#[1,2,7,10,13,15,16,17,18,19,25,29,30],#17
#[3,4,5,6,7,9,10,16,17,18,19,25,26,27,28,29,30,31,32],#18
#[1,2,3,4,7,9,10,15,16,17,18,19,20,23,24,25,26,27,28,29,31],#19
#[3,4,5,6,15,16,17,18,19,21,25,26,27,28,29],#20
#[1,2,3,4,13,15,18,24,25,26,28,29,30],#21
#[3,4,6,7,9,10,13,15,17,18,19,21,23,25,26,27,28,29,30,31,32],#22
#[1,2,3,5,7,9,10,13,15,17,18,19,21,23,24,25,26,27,28,29,30,32]]#23
#

workchns=[
[3,4,6,15,16,17,18,19,21,25,26,27,29,30],#0
[1,2,3,13,16,19,25,26,29],#1
[4,7,9,10,15,16,17,18,19,20,21,25,27,28,29,30,31,32],#2
[1,2,7,9,10,13,15,18,19,21,23,24,25,26,28,29,30],#3
[3,4,5,13,16,17,27,28,29,30,31],#4
[1,2,3,4,5,13,20,24,25,26,27],#5
[15,17,18,19,21,25,26,27,28,29,30,31,32],#6
[4,23,24,25,26,27,28],#7
[3,4,5,6,7,9,10,15,16,17,18,19,24,25,26,27,28,29,30],#8
[1,2,4,7,9,10,15,16,17,18,19,25,27,31],#9
[3,18,19,23,24,25,26,27,28,30],#10
[1,3,4,5,6,16,17,18,19,20,21,27,30,31,32],#11
[3,4,5,6,7,9,10,13,15,17,18,19,21,23,24,25,26,27,28,29,30,31,32],#12
[1,2,3,5,6,7,9,10,13,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32],#13
[5,9,10,13,17,19,20,21,23,24,25,26,27,28,29,30,31,32],#14
[1,2,3,4,6,7,9,13,15,16,17,18,20,23,24,25,26,27,30,31,32],#15
[3,4,7,10,13,15,16,17,18,19,20,21,25,26,29,30,31],#16
[1,2,7,10,13,15,16,17,18,19,25,29,30],#17
[3,4,5,6,7,9,10,16,17,18,19,20,25,26,27,28,29,30,31,32],#18
[1,2,3,4,7,9,10,15,16,17,18,19,20,23,24,25,26,27,28,29,31],#19
[3,4,5,6,15,16,17,18,19,21,25,26,27,28,29],#20
[1,2,3,4,13,15,18,24,25,26,28,29,30],#21
[3,4,6,7,9,10,13,15,17,18,19,21,23,25,26,27,28,29,30,31,32],#22
[1,2,3,5,6,7,9,10,13,15,17,18,19,21,23,24,25,26,27,28,29,30,32]]#23


date = '20200124'
datan= 'biasscan_R1p11_onmount_sky_biasX'

filename='/home/czhang/analysispy/tespy/output/%s/%s/biasscan_filtlinear_nfft10000_at10.0/biasscan_Np10.0.pkl'%(date,datan) #plt on 20min
d=pkl.load(open(filename,"rb"))


biass=d[1][:,0]
biass_=biass.copy()
biass.sort()

for im in range(12):
    spss=np.full([len(d[2]),2*len(d[2][0][0]),len(d[2][0][0][0])], np.nan)
    spss_=np.full([len(d[2]),2*len(d[2][0][0]),len(d[2][0][0][0])], np.nan)
    for ib in range(len(d[1])):
        ib_=np.where(biass_==biass[ib])[0][0]
        spss[ib,:,:]=d[2][ib_][2*im:2*im+2].reshape(2*len(d[2][0][0]),len(d[2][0][0][0]))*1e12
        spss_[ib,:,:]=d[2][ib_][2*im:2*im+2].reshape(2*len(d[2][0][0]),len(d[2][0][0][0]))*1e12
        for row in range(33):
            if not (row in workchns[2*im]):
                spss[ib,row,:]=np.full(len(d[2][0][0][0]), np.nan)
            if not (row in workchns[2*im+1]):
                spss[ib,row+33,:]=np.full(len(d[2][0][0][0]), np.nan)
    spss=spss.reshape(len(d[2])*2*len(d[2][0][0]), len(d[2][0][0][0]))
    spss_=spss_.reshape(len(d[2])*2*len(d[2][0][0]), len(d[2][0][0][0]))
   

    plt.figure()
    spssm=np.ma.masked_where(np.isnan(spss),spss)
    plt.pcolormesh(np.linspace(210.44/2/len(spss[0]), 210.44/2, len(spss[0])), np.array(range(len(d[2])*2*len(d[2][0][0]))), spssm,vmin=10, vmax=300, cmap='coolwarm')
    plt.colorbar().set_label('pA/rtHz')
    for ii in range(len(d[2])):
        bias=int(biass[ii])
        plt.axhline(y=ii*2*len(d[2][0][0]), color='k')
        plt.text(12, ii*66+15, 'bias %d'%bias, fontsize='16')
    plt.ylabel('chnl(66) X bias', fontsize=14)
    plt.xlabel('Hz', fontsize=14)
    plt.xscale('log')
    plt.xlim(0.04,100)
    plt.title('%s'%modules[im], fontsize=20)
    plt.savefig('/home/czhang/analysispy/tespy/output/%s/%s/ps_linear_biasX_%s.png'%(date,datan,modules[im]))	

    plt.figure()
    spssm=np.ma.masked_where(np.isnan(spss),spss)
    plt.pcolormesh(np.linspace(210.44/2/len(spss[0]), 210.44/2, len(spss[0])), np.array(range(len(d[2])*2*len(d[2][0][0]))), np.log10(spssm), vmin=1, vmax=3, cmap='coolwarm')
    plt.colorbar().set_label('log10 pA/rtHz')
    for ii in range(len(d[2])):
        bias=int(biass[ii])
        plt.axhline(y=ii*2*len(d[2][0][0]), color='k')
        plt.text(12, ii*66+15, 'bias %d'%bias, fontsize='16')
    plt.ylabel('chnl(66) X bias', fontsize=14)
    plt.xlabel('Hz', fontsize=14)
    plt.xscale('log')
    plt.xlim(0.04,100)
    plt.title('%s'%modules[im], fontsize=20)
    plt.savefig('/home/czhang/analysispy/tespy/output/%s/%s/ps_log_biasX_%s.png'%(date,datan,modules[im]))	
