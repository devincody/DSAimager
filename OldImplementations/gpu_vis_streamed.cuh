#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>

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


// int call_copy_if (cufftComplex* data, int size, float* value, float* index, float *n_pts){
// 	int ans = thrust::copy_if(thrust::cuda::par.on(streams[i]),
// 				make_zip_iterator(make_tuple(data, make_counting_iterator(0u))),
// 				make_zip_iterator(make_tuple(data, make_counting_iterator(0u))) + size,
// 				data,
// 				make_zip_iterator(make_tuple(value.begin(), index.begin())),
// 				greater_than_val<float>(thresh)
// 				) - make_zip_iterator(make_tuple(value.begin(), index.begin()));
// 	return ans;
// }

__global__
void real(cufftComplex* C, int size){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	while (tid < size){
		C[tid].y = 0;
		tid += gridDim.x * blockDim.x;
	}
}

__global__
void grid(cufftComplex* A, float* data, cufftComplex* A_cal, cufftComplex* B_cal, int idx_chunk_start, int idx_chunk_end){
	/*
	d_A is size SZ*SZ*sizeof(cufftComplex)
	d_data is size 220*1500*sizeof(float)
	d_n_cal is size 45*1500*sizeof(cufftComplex)	
	start and stop are between 0 and usable_frequencies
	*/

	int baseline_number = blockDim.x * blockIdx.x + threadIdx.x; // since baseline is fastest index in data
																 // enables coalesced memory accesses
	
	// Convert from Triangular linear index to (i,j) matrix indicies
	// Derived using polynomial fits
	int i = floor((19.0 - sqrt(361.0 - 8.0*baseline_number))/2.0); // equvalent to sm
	int j = baseline_number - static_cast<int>(-i*i/2.0 + 8.5*i - 1); // larger number

	// printf("(%d, %d, %d)", baseline_number, i, j);

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
	int ant_order[] = {1, 4, 5, 8, 6, 9, 2, 10, 3, 7};
	float pos_extent =  1299.54002797;

	float c = 299792458.0;
	float wavelength;
	int x, y;

	// to be passed as arguments later
	int 	idx_usable_start 	= 300;
	int 	idx_usable_end 	 	= 1800;
	float 	f_tot_start 		= 1.23;
	float 	f_tot_end 			= 1.58;
	int 	n_chunk_idx 		= idx_chunk_end - idx_chunk_start; 	// number of chunk  frequency indicies
	int 	n_usable_idx 		= idx_usable_end - idx_usable_start; 	// number of usable frequency indicies
	int 	n_tot_idx 			= 2048;	 								// number of total  frequency indicies

	float bw_per_channel = (f_tot_end - f_tot_start)/static_cast<float>(n_tot_idx);

	int px_per_res_elem = 3;
	float mv = ((float) SZ)/(2.0*pos_extent)* c/(f_tot_end*1E9)/px_per_res_elem;
	
	cufftComplex tempa, tempb, tempc;

	for (int f = idx_chunk_start; f < idx_chunk_end; f++){
		// In this simulation, we are analyzing the top 250 frequencies
		wavelength = c/(1E9*(f_tot_end - static_cast<float>(f + idx_usable_start)*bw_per_channel ));
		
		// load the A pol data into memory
		tempa.x = data[f*220 + 4*baseline_number];
		tempa.y = data[f*220 + 4*baseline_number+1];


		// load the B pol data into memory
		tempb.x = data[f*220 + 4*baseline_number+2];
		tempb.y = data[f*220 + 4*baseline_number+3];

		tempc.x = tempa.x;
		tempc.y = tempa.y;

		// correct each baseline
		cx_mult(tempa, A_cal[baseline_number*n_usable_idx + f]);
		cx_mult(tempb, B_cal[baseline_number*n_usable_idx + f]);

		// printf("baseline_number*n_usable_idx + f = %d => %f + %fj   vs.  %f + %fj\n", baseline_number*n_usable_idx + f, tempa.x, tempa.y, tempc.x, tempc.y);
		// printf("(%d, %d) => %f + %fj   vs.  %f + %fj\n", i, j , tempa.x, tempa.y, tempc.x, tempc.y);

		//atomic add to baseline 1
		x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength));
		y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength));
		atomicAdd(&(A[x*SZ+y].x),  tempa.x + tempb.x);
		atomicAdd(&(A[x*SZ+y].y), -tempa.y - tempb.y);
		//atomic add to baseline 2
		x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[j]-1][0]-pos[ant_order[i]-1][0])/wavelength));
		y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[j]-1][1]-pos[ant_order[i]-1][1])/wavelength));
		atomicAdd(&(A[x*SZ+y].x),  tempa.x + tempb.x);
		atomicAdd(&(A[x*SZ+y].y),  tempa.y + tempb.y);
	}
}