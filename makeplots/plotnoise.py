import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt

modules=['N2','M4','N3','M5','M6','N1','M8','Mx2','M3','M7','N4','M9']
colors =['orange','yellow','limegreen','green','cyan','skyblue','blue','grey','b','salmon','pink','grey']

##plt on
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


workchns=[
[3,4,6,15,16,17,18,19,21,25,26,27,29,30],#0
[1,2,3,13,16,19,25],#1
[7,9,10,15,16,17,18,19,20,25,27,28,29,30,31],#2
[1,2,7,9,10,13,15,18,19,23,24,25,26,28,29,30],#3
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

for ib in range(len(d[0])):
    bias=int(d[1][ib][0])
    spss=d[2][ib]*1e12
    spss_=d[2][ib]*1e12
    spss=spss.reshape(24*33, len(spss[0][0]))
    spss_=spss_.reshape(24*33, len(spss_[0][0]))
    for col in range(24):
        for row in range(33):
            if not (row in workchns[col]):
                spss[col*33+row]=np.full(len(spss[0]), np.nan)
    
    plt.figure()
    spssm=np.ma.masked_where(np.isnan(spss),spss)
    plt.pcolormesh(np.linspace(210.44/2/len(spss[0]), 210.44/2, len(spss[0])), np.array(range(33*24)), np.log10(spssm),vmin=1, vmax=3, cmap='coolwarm')
    #plt.scatter(np.array([np.linspace(210.44/2/len(spss[0]), 210.44/2, len(spss[0]))]*24*33).reshape(len(spss[0])*24*33),
    #            np.array([[det for ii in range(len(spss[0]))] for det in range(24*33)]).reshape(len(spss[0])*24*33),
    #            c=spss.reshape(len(spss[0])*24*33))
    plt.colorbar().set_label('log10 pA/rtHz')
    for ii in range(12):
        plt.axhline(y=ii*66, color='k')
        plt.text(25, ii*66+15, '%s'%modules[ii], fontsize='25')
    plt.ylabel('gcp', fontsize=14)
    plt.xlabel('Hz', fontsize=14)
    plt.xscale('log')
    plt.xlim(0.04,100)
    plt.savefig('/home/czhang/analysispy/tespy/output/%s/%s/ps_log_workchns_bias%d.png'%(date,datan,bias))	
    
    
    f = np.linspace(210.44/2/len(spss[0]), 210.44/2, len(spss[0]))
    plt.figure()
    for im in range(12):
        plt.clf()
        for col in [im*2,im*2+1]:
            for row in range(33):
                plt.plot(f,spssm[col*33+row], linewidth=0.2, color='b')
        plt.plot(f,np.nanmean(spssm[im*2*33:im*2*33+66],0), linewidth=1, color='r')
        plt.xlabel('Hz', fontsize=14)
        plt.ylabel('pA/rtHz', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.04, 100)
        plt.ylim(10, 10000)
        plt.grid()
        plt.title('%s, bias %d'%(modules[im], bias), fontsize=14)
        plt.savefig('/home/czhang/analysispy/tespy/output/%s/%s/ps_log_%s_bias%d.png'%(date,datan,modules[im], bias))
    
    plt.figure()
    for im in range(12):
        plt.clf()
        for col in [im*2,im*2+1]:
            for row in [0,11,22]:
                plt.plot(f,spss_[col*33+row], linewidth=1, color='b')
        #plt.plot(f,np.nanmean(spssm[im*2*33:im*2*33+66],0), linewidth=1, color='r')
        plt.xlabel('Hz', fontsize=14)
        plt.ylabel('pA/rtHz', fontsize=14)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.04, 100)
        plt.ylim(10, 10000)
        plt.grid()
        plt.title('dark squids in %s, bias %d'%(modules[im], bias), fontsize=14)
        plt.savefig('/home/czhang/analysispy/tespy/output/%s/%s/ps_log_dss_%s_bias%d.png'%(date,datan,modules[im], bias))
    
    
    plt.clf()
    dsind=[0 for ii in range(24*3)]
    ii=0
    for im in range(12):
        for col in [im*2,im*2+1]:
            for row in [0,11,22]:
                plt.plot(f,spss_[col*33+row], linewidth=0.2, color='b')
                dsind[ii]=col*33+row
                ii+=1
    plt.plot(f,np.nanmean(spss_[dsind],0), linewidth=1, color='r')
    print(min(f))
    print(min(np.nanmean(spss_[dsind],0)))
    plt.xlabel('Hz', fontsize=14)
    plt.ylabel('pA/rtHz', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.04, 100)
    plt.ylim(10, 10000)
    plt.grid()
    plt.title('dark squids, bias %d'%bias, fontsize=14)
    plt.savefig('/home/czhang/analysispy/tespy/output/%s/%s/ps_log_dss_bias%d.png'%(date,datan,bias))
