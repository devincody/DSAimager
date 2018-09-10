fact_x = np.zeros((45,1500), dtype = np.complex64)
fact_y = np.zeros((45,1500), dtype = np.complex64)

for i in range(nants*(nants-1)/2):
	fact_x[i,:]=np.exp(-1j*PI2*f*bline_delay_x[i]-1j*phase_x[i,:]).astype(np.complex64)
	fact_y[i,:]=np.exp(-1j*PI2*f*bline_delay_y[i]-1j*phase_y[i,:]).astype(np.complex64)


np.savetxt("AA_real.txt", np.real(fact_x))
np.savetxt("AA_imag.txt", np.imag(fact_x))
np.savetxt("BB_real.txt", np.real(fact_y))
np.savetxt("BB_imag.txt", np.imag(fact_y))