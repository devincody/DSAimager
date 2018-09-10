import sys
import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt


def interp_phase(d,FREQ1,NF1, bsf=30):
	maxf=np.amax(FREQ1)
	f_cal=FREQ1[bsf/2::bsf]/1e9
	dr=np.zeros((d.shape[0],NF1),dtype=np.float32)
	di=np.zeros((d.shape[0],NF1),dtype=np.float32)
	for i in range(d.shape[0]):
		dr[i,:]=np.interp(maxf-FREQ1/1e9, maxf-f_cal, d[i,:].real)
		di[i,:]=np.interp(maxf-FREQ1/1e9, maxf-f_cal, d[i,:].imag)
	return dr+1j*di


# MAIN #

def applyCal(cal_fname_x="Acal.npz",cal_fname_y="Bcal.npz",vis_fname="../test.fits",burst_loc=231):

	# frequency definitions
	FREQ=1e6*(1530.0-np.arange(0.0,2048.0,1.0)*250.0/2048.0)
	F_START=300
	F_END=1800
	FREQ1=FREQ[F_START:F_END]
	NF1=len(FREQ1)
	PI2 = 2.*np.pi

	# read A cal (x)
	tp=np.load(cal_fname_x)
	anames_x=list(tp['aname']) # antenna ordering
	taus_x=tp['ant_delays'] # delays (s) for each antenna
	g_x = tp['gains'] # complex gains - only worry about phase
	bsf_x=1500/tp['bsf']
	g_x=np.nanmean(g_x[:,g_x.shape[1]/2-2:g_x.shape[1]/2+2,:],axis=1)
	
	# read B cal (y)
	tp=np.load(cal_fname_y)
	anames_y=list(tp['aname'])
	taus_y=tp['ant_delays']
	g_y=tp['gains']
	bsf_y=1500/tp['bsf']
	g_y=np.nanmean(g_y[:,g_y.shape[1]/2-2:g_y.shape[1]/2+2,:],axis=1)
	
	# because we only calibrate on frequncy-averaged data, interpolate to full resolution
	g_x_interp = interp_phase(g_x, FREQ1, NF1)#, bsf_x)
	g_y_interp = interp_phase(g_y, FREQ1, NF1)#, bsf_y)
	
	# open visibility data header
	f=FREQ1
	h = pf.open(vis_fname)[1].header
	antennas = h['ANTENNAS'].split('-')
	nants = len(antennas)
	
	# define baseline delays and gains.
	bline_delay_x=np.zeros((nants*(nants-1)/2),dtype=np.float32)
	bline_delay_y=np.zeros((nants*(nants-1)/2),dtype=np.float32)
	phase_x=np.zeros((nants*(nants-1)/2,NF1),dtype=np.float32)
	phase_y=np.zeros((nants*(nants-1)/2,NF1),dtype=np.float32)
	
	iiter=0
	for i in range(nants-1):
	    for j in range(i+1,nants):
	        a1=int(antennas[i])-1; a2=int(antennas[j])-1;
	        # baseline delays and gains for pol x (=A)
	        t1=taus_x[a1]
	        t2=taus_x[a2]
	        g1=g_x_interp[a1,:]
	        g2=g_x_interp[a2,:]
	        phase_x[iiter,:] = np.angle(g1*np.conjugate(g2))
	        bline_delay_x[iiter]=t1-t2
	        # baseline delays and gains for pol y (=B) 
	        t1=taus_y[a1]
	        t2=taus_y[a2]
	        bline_delay_y[iiter]=t1-t2
	        g1=g_y_interp[a1,:]
	        g2=g_y_interp[a2,:]
	        phase_y[iiter,:] = np.angle(g1*np.conjugate(g2))
	        iiter+=1
	

	#open visiblity data data
	d=pf.open(vis_fname)[1].data['Data'].astype(np.float32)
	header = pf.open(vis_fname)[1].header
	nt = (header['NAXIS2']-1)/(2048*220)
	print("There are %d time-samples in visibility data"%nt)
	d1=d[:nt*2048*220]; 
	dat = np.reshape(d1,newshape=(nt,2048,220)); 
	
	# prepare cross correlations
	burst_nsamp = 1

	cc = np.mean(dat[burst_loc:burst_loc+burst_nsamp,F_START:F_END,:180],axis=0) # cross correlations (1500,180)
	cc1=cc[:,::2]+1j*cc[:,1::2]
	cc1[np.where(cc1.real==0.0)]=np.nan+1j*np.nan
	cc_xx=cc1[:,::2]; 
	cc_yy=cc1[:,1::2]
	
	# output vis data for burst_loc time sample
	vis = np.zeros((NF1,nants*(nants-1)/2),dtype=np.complex64)
	
	# calibrate!
	for i in range(nants*(nants-1)/2):
		fac_x=np.exp(-1j*PI2*f*bline_delay_x[i]-1j*phase_x[i,:]).astype(np.complex64)
		fac_y=np.exp(-1j*PI2*f*bline_delay_y[i]-1j*phase_y[i,:]).astype(np.complex64)
		vis[:,i] =( (cc[:,4*i]+1j*cc[:,4*i+1])*fac_x  + (cc[:,4*i+2]+1j*cc[:,4*i+3])*fac_y  ) /2.0 #average vis, apply cal

	return vis



def main():
	vis = applyCal()
	plt.plot(vis)
	plt.show()
	print vis.shape

if __name__ == '__main__':
	main()