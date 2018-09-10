import numpy as np      
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D
from antennaLocations import get_pos


use_visibilities = True
use_DSA_positions = True

print '#-'*20 + "#"
if use_visibilities:
	print("USING VISIBILITIES")
else:
	print("USING VOLTAGES")
print '#-'*20 + "#"

def z(a,b):
    return np.sqrt(1-a**2-b**2)


np.random.seed(220)

samples = 2048

t = np.linspace(0, 0.000008192, samples) # seconds
c = 3E8
freq_units = 2*np.pi*1E9# Giga-rad/s

# pos_val = np.linalg.norm(pos, axis = 1)
# idx = pos_val.argsort()
# pos = pos[idx]
# std_pos = 1*np.std(pos)

wx = -.01
wy = .012
# if use_visibilities:
# 	antennas = 10
# 	wx /=2
# 	wy /=2
wave = np.array([wx, wy, z(wx, wy)]) # source location along antenna axes



if use_DSA_positions:
	pos = get_pos()
	antennas = len(pos)
	# pos_extent = 1.1*max(max(pos[:,0])-min(pos[:,0]), max(pos[:,1])-min(pos[:,1]))
else:
	antennas = 10
	pos = np.random.rand(antennas,3)*1000
	pos -= np.median(pos, axis = 0)
	pos[:,2] /= 300000
	# pos_extent = 1000
pos_extent = 1.1*max(max(pos[:,0])-min(pos[:,0]), max(pos[:,1])-min(pos[:,1]))


# print '#'*50
# print "plane wave wavenumbers = ", wave
# print '#'*50

# print std_pos, pos



								  # data from dsa => 512 time steps, 4ms
								  # now, 200 time steps => 16 ms
noise = 1*np.random.rand(antennas,samples) + 1.0j*np.random.rand(antennas,samples) 
phase = 1*np.pi/2.342
amp = .5


signal  = 1.0 * amp * np.exp(1.0j*(np.outer(np.dot(pos, wave)*1.2*freq_units/c,np.ones((1,samples))) + 1.2*freq_units*np.outer(np.ones((1,antennas)),t) + phase))
signal += 0.3 * amp * np.exp(1.0j*(np.outer(np.dot(pos, wave)*1.15*freq_units/c,np.ones((1,samples))) + 1.15*freq_units*np.outer(np.ones((1,antennas)),t) + phase))
signal += 1.0 * amp * np.exp(1.0j*(np.outer(np.dot(pos, wave)*1.0*freq_units/c,np.ones((1,samples))) + 1.0*freq_units*np.outer(np.ones((1,antennas)),t) + phase))

for i in np.random.rand(500)*0.25+1:
	signal += 1.0*np.random.rand() * np.exp(1.0j*(np.outer(np.dot(pos, wave)*i*freq_units/c,np.ones((1,samples))) + i*freq_units*np.outer(np.ones((1,antennas)),t) + phase))


# Time series data
summe = signal + noise
# Channelize data
f = np.fft.fft(summe)
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



if 1: # Show serial data and FFT data
	plt.subplot(2,1,1)
	plt.plot(t*1E6, summe.T)
	plt.ylabel("Amplitude")
	plt.xlabel("Time ($\mu s$)")
	plt.subplot(2,1,2)
	plt.plot(np.linspace(1.0, 1.250, 2048), (np.abs(f)**2).T)
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

sz = 512
A = np.zeros((sz,sz), dtype=np.complex)



if not use_visibilities: # Use the voltage "backend"
	mv = float(sz)/float(pos_extent)*c/(1.25E9) #mapping variable
	print("mv = {}".format(mv))
	#smallest wavenumber. calculated by inverse of (width of A grid. i.e. 1000meters/.24meters)
	sm_k = (c/1.25E9)/float(pos_extent)

	for i in range(antennas):
		for j in range(samples): #iterate through frequencies
			wavelength = c/(1E9*(1.0+0.25/samples*j))
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

	mv = float(sz)/float(2*pos_extent)*c/(1.25E9) #mapping variable
	print("mv = {}".format(mv))
	#smallest wavenumber. calculated by inverse of (width of A grid. i.e. 1000meters/.24meters)
	sm_k = (c/1.25E9)/float(2*pos_extent)

	for i in range(antennas):
		for k in range(antennas):
			for j in range(samples): #iterate through frequencies
				wavelength = c/(1E9*(1.0+0.25/samples*j))
				x = int(sz/2 + np.floor(mv*(pos[i,0]-pos[k,0])/wavelength))
				y = int(sz/2 + np.floor(mv*(pos[i,1]-pos[k,1])/wavelength))
				# print x,y

				# if np.abs(f[i,j]*f[k,j]) > np.abs(A[x,y]):
				A[x,y] += f[i,j]*np.conj(f[k,j])*np.abs(f[i,j]*np.conj(f[k,j]))

				numb[x, y] += np.abs(f[i,j]*np.conj(f[k,j]))

	numb[np.where(numb == 0)] = 1#numb[np.where(numb = 0)] 
	A /= numb
	# print(A)



plt.figure()
plt.subplot(1,2,1)
B = np.ma.masked_where(A == 0, A)
cmap = plt.cm.magma
cmap.set_bad(color='white')
plt.imshow(np.abs(B))
plt.xlabel("$b_x/\lambda$")
plt.ylabel("$b_y/\lambda$")
if use_visibilities:
	plt.title("Visibility-space")
else:
	plt.title("Location-space")

plt.subplot(1,2,2)
C = np.abs(np.fft.fft2(A))**2
# temp = C[:,:sz/2]
# C[:,:sz/2] = C[:,sz/2:]
temp = np.copy(C[:sz/2,:])
C[:sz/2,:] = C[sz/2:,:]
C[sz/2:,:] = temp

temp = np.copy(C[:,:sz/2])
C[:,:sz/2] = C[:,sz/2:]
C[:,sz/2:] = temp




plt.imshow(C, cmap = "hot", extent=[-float(sz)/2*sm_k,float(sz)/2*sm_k,float(sz)/2*sm_k,-float(sz)/2*sm_k])
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.title("Point source localization (Actual: $k_x = ${}, $k_y$ = {})".format(wy,wx))

# plt.subplot(1,3,3)
# plt.imshow(np.abs(np.fft.ifft2(A))**2)


loc = np.argmax(C)
print("Predicted location: k_x = {}, k_y = {})".format(sm_k*(loc%sz-sz/2), sm_k*(np.floor(loc/sz)-sz/2)))

plt.show()




