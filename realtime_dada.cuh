#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>

/* dada includes */
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>

#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"


#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 1024													//Size of the 2D FFT
#define N_ANTENNAS 10											//Number of Antennas (10)
#define N_BASELINES ((N_ANTENNAS - 1)*(N_ANTENNAS)/2)			//Number of Baselines (45)
#define N_CORRELATION_PRODUCTS (N_ANTENNAS+N_BASELINES)			//Number of Correlation Products (55)
#define N_STREAMS 5												//Number of Streams
#define N_TIMESTEPS_per_BLOCK 384								//How many timesteps in each block
#define N_BLOCKS_on_GPU 4										//Number of blocks on GPU
#define N_TIMESTEPS (N_TIMESTEPS_per_BLOCK * N_BLOCKS_on_GPU)	//How much data to hold in GPU
#define N_DUMPPOINTS 1000										//How many points to identify in copy_if

// #define GPU 1													//Which GPU? 0-5
#define N_GPUS 5												//Total Number of GPUs
#define F_AVERAGING 2											//Number of channels averaged into each channel

#define ZERO_PT 300												//Start and end indexes are offset by 300 channels from absolute channel number
#define N_CHANNELS_per_GPU 250									//Number of physical (not averaged) channels on each GPU
// #define start_idx (GPU*N_CHANNELS_per_GPU)						//Start index for GPU
// #define end_idx ((GPU+1)*N_CHANNELS_per_GPU)					//Stop index for GPU


#define START_F 1.28											//Start frequency in GHz
#define END_F 1.53												//Stop  frequency in GHz
#define TOT_CHANNELS 2048										//Total Number of channels

#define C_SPEED 299792458.0										//Speed of light
#define N_POL 2
#define N_COMPLEX 2


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

void gpuFFTchk(int errval){
	if (errval != CUFFT_SUCCESS){
		std::cerr << "Failed FFT call, error code " << errval << std::endl;
	}
}



cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
	gpuErrchk(cudaEventCreate(&start));       \
	gpuErrchk(cudaEventCreate(&stop));        \
	gpuErrchk(cudaEventRecord(start));        \
}

#define STOP_RECORD_TIMER(name) {                           \
	gpuErrchk(cudaEventRecord(stop));                     \
	gpuErrchk(cudaEventSynchronize(stop));                \
	gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
	gpuErrchk(cudaEventDestroy(start));                   \
	gpuErrchk(cudaEventDestroy(stop));                    \
}

/* Usage as defined by dada example code */
void usage()
{
  fprintf (stdout,
	   "dsaX_imager [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -k key [default dada]\n"
	   " -h        print usage\n");
}

/*cleanup as defined by dada example code */
void dsaX_dbgpu_cleanup (dada_hdu_t * in,  multilog_t * log) {
	if (dada_hdu_unlock_read (in) < 0){
		multilog(log, LOG_ERR, "could not unlock read on hdu_in\n");
	}
	dada_hdu_destroy (in);
}


__device__
void cx_mult (cufftComplex &a, cufftComplex b){
	cufftComplex temp;
	temp.x = a.x*b.x-a.y*b.y;
	temp.y = a.x*b.y + a.y*b.x;
	a.x = temp.x; a.y = temp.y;
}

template<typename T>
struct greater_than_val : public thrust::unary_function<T, bool> {
	T val;
	greater_than_val(T val_) : val(val_) {}
	inline __host__ __device__
	bool operator()(T x) const {
	return x.x > val.x;
	}
};

using thrust::make_zip_iterator;
using thrust::make_tuple;
using thrust::make_counting_iterator;

__global__
void grid(cufftComplex* A, float* data, int time_step, int gpu, float DM){
	int start_idx = ((gpu)*N_CHANNELS_per_GPU);
	int end_idx = ((gpu+1)*N_CHANNELS_per_GPU);
	int baseline_number = blockIdx.x;
	// int tid = threadIdx.x;
	// int baseline_number = blockDim.x * blockIdx.x + threadIdx.x; // since baseline is fastest index in data
																 // enables coalesced memory accesses
	
	/*
	Convert from Triangular linear index to (i,j) matrix indicies
	Derived using polynomial fits
	*/
	// int i = floor((19.0 - sqrt(361.0 - 8.0*baseline_number))/2.0); // equvalent to sm
	// int j = baseline_number - static_cast<int>(-i*i/2.0 + 8.5*i - 1); // larger number
	int j = floor((1.0 + sqrt(8.0*baseline_number + 1.0))/2.0);
	int i = baseline_number - (j * j - j)/2;
	int baseline_idx = baseline_number + j; //How to get index for data matrix from baseline number
											//Shifting triangular index up by 1
	// printf("(i,j) = (%d, %d), baseline_number = %d, baseline_idx = %d\n", i,j,baseline_number, baseline_idx);

	/* Antenna Locations, Generated by AntennaLocations.py */
	float pos[N_ANTENNAS][3] = {{  2.32700165e-05,   4.43329978e+02,   1.02765904e-05},
							    { 2.21098953e-05,   4.36579978e+02,   9.64180626e-06},
						        { 1.14039406e-05,   3.42179978e+02,   4.94604756e-06},
						        {-1.14039406e-05,   1.53679978e+02,  -4.94604756e-06},
						        { 1.90319971e+02,  -4.48715200e-05,   1.72737512e-05},
						        { 1.83569971e+02,  -4.37387873e-05,   1.57832354e-05},
						        {-1.03180029e+02,  -1.03154892e-05,  -2.81970909e-05},
						        {-1.76180029e+02,  -1.63615256e-06,  -3.96178686e-05},
						        {-2.03330030e+02,   1.63615256e-06,  -4.39237587e-05},
						        {-9.91080054e+02,  -2.16759905e+02,  -4.00017642e+00}};
	// int ant_order[] = {1, 4, 5, 8, 6, 9, 2, 10, 3, 7};
	int ant_order[] = {3, 7, 2, 10, 1, 4, 5, 8, 6, 9};

	float pos_extent =  1299.54002797;

	float wavelength;
	int x_pos, y_pos;

	// to be passed as arguments later
	// int 	idx_usable_start 	= 300;// the start of the "usable" frequencies is the 300th total channel
	// int 	idx_usable_end 	 	= 1800;
	// float 	f_tot_start 		= 1.23;
	// float 	f_tot_end 			= 1.58;
	//int 	n_chunk_idx 		= idx_chunk_end - idx_chunk_start; 	// number of chunk  frequency indicies
	// int 	n_usable_idx 		= idx_usable_end - idx_usable_start; 	// number of usable frequency indicies
	// int 	n_tot_idx 			= 2048;	 								// number of total  frequency indicies

	int n_channels_in_data = (1.0*N_CHANNELS_per_GPU)/F_AVERAGING;  			// Number of channels present in data
	int n_floats_per_baseline = N_POL*N_COMPLEX*n_channels_in_data; 			// Number of floats between baseline indicies
	float bw_per_channel = (START_F - END_F)/static_cast<float>(TOT_CHANNELS);	// Bandwidth in Each channel
	float time_per_step = 1.048576; 											// ms

	float f_low = (END_F - (ZERO_PT + end_idx) * bw_per_channel); 				// Lowest Frequency in dataset
	float time_delay_low_freq = 4.148*DM/(f_low*f_low); 						// DM time delay for lowest frequency (max delay)

	int px_per_res_elem = 3;
	float UV_to_index = ((float) SZ)/(1.0*pos_extent)* C_SPEED/(END_F*1E9)/px_per_res_elem; //convert U,V to index
	
	cufftComplex A_POL_DATUM, B_POL_DATUM;										// Data variables for real, imag
	float f_ghz;																// Frequency of analysis
	float avg_mid_pt = (F_AVERAGING - 1.0)/2.0; 								// Frequency Center of averaged band 
	int DM_offset_index, DM_offset_time_steps;

	int f = threadIdx.x; //index of the frequency which this thread is analyzing

	//float f_hig = (END_F - (ZERO_PT + start_idx) * bw_per_channel); // chunk start <=> high freq
	// int n_floats_per_freq = (N_BASELINES + N_ANTENNAS)*n_pol*n_complex; // number of floats per frequency (220)

	while (f < n_channels_in_data){ //we have data for N_CHANNELS_per_GPU/F_AVERAGING frequencies

		/* Calculate Current Frequency from f, the frequency index,
		   Frequencies are calculated backwards from the highest frequency*/
		f_ghz = (END_F - static_cast<float>(f*F_AVERAGING + avg_mid_pt + start_idx + ZERO_PT)*bw_per_channel);

		/* Calculate Dispersion delay between current frequency, lowest frequency */
		DM_offset_time_steps = floor( (time_delay_low_freq - 4.148*DM/(f_ghz*f_ghz)) / time_per_step);
		
		/* Calculate the index offset */
		DM_offset_index = ((time_step - DM_offset_time_steps)%N_TIMESTEPS) * n_floats_per_baseline * (N_BASELINES + N_ANTENNAS);


		wavelength = C_SPEED/(1E9*f_ghz);
		
		// load the A pol data into memory
		A_POL_DATUM.x = data[DM_offset_index + (baseline_idx * n_floats_per_baseline) + (f*N_POL*N_COMPLEX)    ];
		A_POL_DATUM.y = data[DM_offset_index + (baseline_idx * n_floats_per_baseline) + (f*N_POL*N_COMPLEX) + 1];


		// load the B pol data into memory
		B_POL_DATUM.x = data[DM_offset_index + (baseline_idx * n_floats_per_baseline) + (f*N_POL*N_COMPLEX) + 2];
		B_POL_DATUM.y = data[DM_offset_index + (baseline_idx * n_floats_per_baseline) + (f*N_POL*N_COMPLEX) + 3];

		// correct each baseline
		// cx_mult(A_POL_DATUM, A_cal[baseline_number*n_usable_idx + f]);
		// cx_mult(B_POL_DATUM, B_cal[baseline_number*n_usable_idx + f]);

		// printf("baseline_number*n_usable_idx + f = %d => %f + %fj   vs.  %f + %fj\n", baseline_number*n_usable_idx + f, A_POL_DATUM.x, A_POL_DATUM.y, tempc.x, tempc.y);
		// printf("(%d, %d) => %f + %fj   vs.  %f + %fj\n", i, j , A_POL_DATUM.x, A_POL_DATUM.y, tempc.x, tempc.y);

		//atomic add to baseline 1
		x_pos = static_cast<int>(floor(UV_to_index*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength)) % SZ;
		y_pos = static_cast<int>(floor(UV_to_index*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength)) % SZ;
		atomicAdd(&(A[x_pos*SZ+y_pos].x),  A_POL_DATUM.x + B_POL_DATUM.x);
		atomicAdd(&(A[x_pos*SZ+y_pos].y), -A_POL_DATUM.y - B_POL_DATUM.y);
		//atomic add to baseline 2
		x_pos = static_cast<int>(floor(UV_to_index*(pos[ant_order[j]-1][0]-pos[ant_order[i]-1][0])/wavelength)) % SZ;
		y_pos = static_cast<int>(floor(UV_to_index*(pos[ant_order[j]-1][1]-pos[ant_order[i]-1][1])/wavelength)) % SZ;
		atomicAdd(&(A[x_pos*SZ+y_pos].x),  A_POL_DATUM.x + B_POL_DATUM.x);
		atomicAdd(&(A[x_pos*SZ+y_pos].y),  A_POL_DATUM.y + B_POL_DATUM.y);

		f += blockDim.x;
	}
}
































