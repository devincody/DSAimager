

import numpy as np      
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D
from antennaLocations import get_pos
import pyfits as pf
import sys
from applyCal import applyCal


def main():
	use_visibilities = True
	use_DSA_positions = True
	use_crab_data = True
	use_voltages_for_vis = False # only intersting when use_crab_data is true
	time_n = 232

	def z(a,b):
		return np.sqrt(1-a**2-b**2)

	f_actual_start = 1.28
	f_actual_end = 1.53
	bw = f_actual_end - f_actual_start

	if use_crab_data:
		use_DSA_positions = True
		f_start = 300; f_end = 1800

		# time_list = range(100,330,3)
		time_list = [229,230, 231,232,233]
	else:
		use_voltages_for_vis = True
		f_start = 0; f_end = 2048
		time_list = [0]
		wx = -.01
		wy = .012
		wave = np.array([wx, wy, z(wx, wy)]) # source location along antenna axes

	np.random.seed(220)

	t_samples = 2048
	f_samples = f_end-f_start
	print("Fsamples = {}".format(f_samples))

	t = np.linspace(0, 0.000008192, t_samples) # seconds
	c = 3E8
	freq_units = 2*np.pi*1E9# Giga-rad/s

	# pos_val = np.linalg.norm(pos, axis = 1)
	# idx = pos_val.argsort()
	# pos = pos[idx]
	# std_pos = 1*np.std(pos)


	# if use_visibilities:
	# 	antennas = 10
	# 	wx /=2
	# 	wy /=2


	if use_DSA_positions:
		pos = get_pos()
		antennas = len(pos)
		print("using {} antennas".format(antennas))
		# pos_extent = 1.1*max(max(pos[:,0])-min(pos[:,0]), max(pos[:,1])-min(pos[:,1]))
	else:
		antennas = 10
		pos = np.random.rand(antennas,3)*1000
		pos -= np.median(pos, axis = 0)
		pos[:,2] /= 300000
		# pos_extent = 1000
	pos_extent = 1.1*max(max(pos[:,0])-min(pos[:,0]), max(pos[:,1])-min(pos[:,1]))


	for time_n in time_list:
		print '#'*30 + "#"
		if use_visibilities:
			print("USING VISIBILITIES")
		else:
			print("USING VOLTAGES")
		print '#'*30 + "#"

										  # data from dsa => 512 time steps, 4ms
										  # now, 200 time steps => 16 ms

		if (not use_crab_data):
			noise = 1*np.random.rand(antennas,t_samples) + 1.0j*np.random.rand(antennas,t_samples) 
			phase = 1*np.pi/2.342
			amp = .5

			signal  = 1.0 * amp * np.exp(1.0j*(np.outer(np.dot(pos, wave)*(f_actual_start+.2)*freq_units/c,np.ones((1,t_samples))) + (f_actual_start+.2)*freq_units*np.outer(np.ones((1,antennas)),t) + phase))
			signal += 0.3 * amp * np.exp(1.0j*(np.outer(np.dot(pos, wave)*(f_actual_start + 0.15)*freq_units/c,np.ones((1,t_samples))) + (f_actual_start + 0.15)*freq_units*np.outer(np.ones((1,antennas)),t) + phase))
			signal += 1.0 * amp * np.exp(1.0j*(np.outer(np.dot(pos, wave)*(f_actual_start)*freq_units/c,np.ones((1,t_samples))) + (f_actual_start)*freq_units*np.outer(np.ones((1,antennas)),t) + phase))

			for i in np.random.rand(500)*bw+f_actual_start:
				signal += 1.0*np.random.rand() * np.exp(1.0j*(np.outer(np.dot(pos, wave)*i*freq_units/c,np.ones((1,t_samples))) + i*freq_units*np.outer(np.ones((1,antennas)),t) + phase))
			# Time series data
			summe = signal + noise
			# Channelize data
			f = np.fft.fft(summe)
			print f.shape #(10, 2048)
			# f = f[:,f_start:f_end]s
		else:
			file_name = "../test.fits"
			n = 512 # number of time-samples
			print pf.open(file_name)[1].data
			data = np.asarray(pf.open(file_name)[1].data).astype('float')
			data = data[0:len(data)-1]

			print 'TSAMP (s): ',pf.open(file_name)[1].header['TSAMP']
			# print antenna order
			ant_ord = pf.open(file_name)[1].header['ANTENNAS']
			ant_ord = [int(x) for x in ant_ord.split("-")]
			print 'ANTENNA ORDER: ',ant_ord
			# pf.close()
			# data = data.reshape((n,2048,220))
			# data = data[time_n,:,:]
			# print data.shape
			# voltages = data[:,180:]
			# # flag edge channels
			# voltages = voltages[f_start:f_end,:]
			cross_corr = applyCal(burst_loc = time_n)#data[:,:180]
			print("Xcor shape: ", cross_corr.shape)
			# plt.plot(cross_corr)
			# plt.show()
			# cross_corr = cross_corr[f_start:f_end,:]
			# print("Xcor shape: ", cross_corr.shape)

			# print voltages.shape # (1648, 40)

			# f = np.zeros((10,f_samples), dtype = np.complex)
			# for i in range(antennas):
			# 	f[ant_ord[i]-1,:] = voltages[:,2*(i)] + 1j*voltages[:,2*(i)+1]
				# f[i,:] = voltages[:,2*i] + 1j*voltages[:,2*i+1]

				


		# print f

		# Single frequency channel
		if 0: # Calculate single channel power
			test_sinusoid  = np.exp(- 1.2*freq_units*t*1j)
			single_f = np.dot(summe, test_sinusoid) 
			print(np.abs(sum(single_f)))
			test_sinusoid  = np.exp(- 1.2343*freq_units*t*1j)
			single_f = np.dot(summe, test_sinusoid) 
			print(np.abs(sum(single_f)))

			test_sinusoid  = np.exp(- 1.4*freq_units*t*1j)
			single_f = np.dot(summe, test_sinusoid) 
			print(np.abs(sum(single_f)))
			test_sinusoid  = np.exp(- 1.4343*freq_units*t*1j)
			single_f = np.dot(summe, test_sinusoid) 
			print(np.abs(sum(single_f)))



		if 0: # Show serial data and FFT data
			plt.subplot(2,1,1)
			plt.plot(t*1E6, summe.T)
			plt.ylabel("Amplitude")
			plt.xlabel("Time ($\mu s$)")
			plt.subplot(2,1,2)
			print f_actual_start + float(f_start)/t_samples*bw, f_actual_start+float(f_end)/t_samples*bw, f_samples
			plt.plot(np.linspace(f_actual_start + float(f_start)/t_samples*bw, f_actual_start+float(f_end)/t_samples*bw, f_samples), (np.abs(f)**2).T)
			plt.ylabel("Power (arb.)")
			plt.xlabel("Frequency (GHz)")
			plt.show()

		if 0: # Plot antenna locations in 3d and 2d
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(pos[:,0], pos[:,1], pos[:,2])
			ax.set_zlim3d(-pos_extent,pos_extent)
			plt.show()

			plt.scatter(pos[:,0], pos[:,1])
			plt.show()

		sz = 2048
		A = np.zeros((sz,sz), dtype=np.complex)
		px_per_res_elem = 3



		if not use_visibilities: # Use the voltage "backend"
			mv = float(sz)/float(pos_extent)*c/(f_actual_end*1E9)/px_per_res_elem #mapping variable
			print("mv = {}".format(mv))

			#smallest wavenumber. calculated by inverse of (width of A grid. i.e. 1000meters/.24meters)
			sm_k = (c/(f_actual_end*1E9))/float(pos_extent)/px_per_res_elem
			print("sm_k = {}".format(sm_k))

			for i in range(antennas):
				for j in range(f_samples): #iterate through frequencies
					wavelength = c/(1E9*(f_actual_end - float(f_end + j)/t_samples*bw))
					# print "freq = {}".format((1.0 + float(f_start + j)/t_samples*bw))
					x = int(sz/2 + np.floor(mv*pos[i,0]/wavelength))
					y = int(sz/2 + np.floor(mv*pos[i,1]/wavelength))
					# print(wavelength)
					# print(sz/2 + np.floor(mv*pos[i,0]*wavelength))
					# print(sz/2 + np.floor(mv*pos[i,1]*wavelength))
					if np.abs(f[i,j]) > np.abs(A[x,y]):
						A[x,y] = f[i,j]
					# numb[x, y] += 1.0

		if use_visibilities:
			numb = np.zeros((sz,sz))

			mv = float(sz)/float(2*pos_extent)*c/(f_actual_end*1E9)/px_per_res_elem #mapping variable
			print("mv = {}".format(mv))
			print("pos_extent = {}".format(pos_extent))
			print("other = {}".format(c/(f_actual_end*1E9)/px_per_res_elem))
			#smallest wavenumber. calculated by inverse of (width of A grid. i.e. 1000meters/.24meters)
			sm_k = (c/(f_actual_end*1E9))/float(2*pos_extent)/px_per_res_elem
			print("sm_k = {}".format(sm_k))

			if use_voltages_for_vis:
				print("using voltages for visibilities")

			count = 0
			for i in range(antennas):
				for k in range(antennas):
					for j in range(f_samples): #iterate through frequencies
						# wavelength = c/(1E9*(f_actual_end - float(f_end + j)/t_samples*bw))
						wavelength = c/(1E9*(f_actual_end - float(f_start + j)/t_samples*bw))
						
						
						if (i == k):
							continue

						x = int(sz/2 + np.floor(mv*(pos[ant_ord[i]-1,0]-pos[ant_ord[k]-1,0])/wavelength))
						y = int(sz/2 + np.floor(mv*(pos[ant_ord[i]-1,1]-pos[ant_ord[k]-1,1])/wavelength))

						
						if (i < k):
							sm = i; lg = k
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2
							# cor = cross_corr[j,4*idx] - 1j* cross_corr[j, 4*idx+1]
							cor = np.conj(cross_corr[j,idx])
							A[x,y] += cor
							numb[x, y] += np.abs(cor)
						else:
							sm = k; lg = i
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2
							cor = cross_corr[j,idx]
							A[x,y] += cor
							numb[x, y] += np.abs(cor)

						# if x > (sz*.7):
						# 	count += 1
						# 	print("x = {}, y = {}, wav = {}, c = {}".format(x,y, wavelength, count))



			# print("coutn = {}".format(count))

			# print(A)
			# print(numb)
			# for i in range(sz):
			# 	for j in range(sz):
			# 		if (np.abs(A[i,j]) > 1):
			# 			print(A[i,j])
			# 	print("break")			

						

			numb[np.where(numb == 0)] = 1#numb[np.where(numb = 0)] 
			A /= numb
			# print(A)

		plt.figure(figsize = (12,6))
		plt.subplot(1,2,1)
		B = np.ma.masked_where(A == 0, A)
		cmap = plt.cm.magma
		cmap.set_bad(color='white')
		vz_low = sz*(px_per_res_elem-1)/(px_per_res_elem*2)
		vz_high = sz*(px_per_res_elem+1)/(px_per_res_elem*2)

		plt.imshow(np.abs(B[vz_low:vz_high, vz_low:vz_high]))
		# plt.spy(np.abs(B[vz_low:vz_high, vz_low:vz_high]))
		# T = np.zeros((sz,sz))
		# T[np.where(np.abs(B)>0)] == 1
		# plt.imshow(T)

		plt.xlabel("$b_x/\lambda$")
		plt.ylabel("$b_y/\lambda$")
		if use_visibilities:
			plt.title("Visibility-space")
		else:
			plt.title("Location-space")

		plt.subplot(1,2,2)
		for i in range(10):
			C = np.fft.fft2(A)
		C = np.abs(C)**2
		# temp = C[:,:sz/2]
		# C[:,:sz/2] = C[:,sz/2:]
		temp = np.copy(C[:sz/2,:])
		C[:sz/2,:] = C[sz/2:,:]
		C[sz/2:,:] = temp

		temp = np.copy(C[:,:sz/2])
		C[:,:sz/2] = C[:,sz/2:]
		C[:,sz/2:] = temp



		if (not use_visibilities and use_crab_data):
			print("time_n ={}, min = {}, max = {}". format(time_n, np.min(C), np.max(C)))
			plt.imshow(C, cmap = "hot", extent=[-float(sz)/2*sm_k,float(sz)/2*sm_k,float(sz)/2*sm_k,-float(sz)/2*sm_k])
		elif (use_visibilities and use_crab_data):
			print("time_n ={}, min = {}, max = {}". format(time_n, np.min(C), np.max(C)))
			ext = [-float(sz)/2*sm_k,float(sz)/2*sm_k,float(sz)/2*sm_k,-float(sz)/2*sm_k]
			# ext = None
			plt.imshow(C, cmap = "hot", extent = ext, vmax = 50000)#np.max(C))
		else:
			print("min = {}, max = {}". format(np.min(C), np.max(C)))
			plt.imshow(C, cmap = "hot", extent=[-float(sz)/2*sm_k,float(sz)/2*sm_k,float(sz)/2*sm_k,-float(sz)/2*sm_k])

		plt.xlabel("$k_x$")
		plt.ylabel("$k_y$")

		if use_crab_data:
			plt.title("Crab Nebula Pulsar")
			plt.colorbar()
			plt.savefig("frames4/" + str(time_n).zfill(4) + ".png")
			# plt.show()
			plt.close()
		else:
			plt.title("Point source localization (Actual: $k_x = ${}, $k_y$ = {})".format(wy,wx))
			loc = np.argmax(C)
			print("Predicted location: k_x = {}, k_y = {})".format(sm_k*(loc%sz-sz/2), sm_k*(np.floor(loc/sz)-sz/2)))
			print("\t\t(Actual: k_x = {}, k_y = {})".format(wy,wx))
			plt.show()

		# plt.subplot(1,3,3)
		# plt.imshow(np.abs(np.fft.ifft2(A))**2)

		
if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("USER ENDED!")
		sys.exit()



