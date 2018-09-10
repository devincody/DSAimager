#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>
// #include <thrust/max_element.h>
#include <iostream>
#include <cstdio>
#include <curand.h>

//nvcc -o thrust_max thrust_max_test.cu -lcurand -Wno-deprecated-gpu-targets

#define N 1024
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

void GPU_fill_rand(float *A, int nr_rows, int nr_cols){
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	curandGenerateUniform(prng, A, nr_rows*nr_cols);
}

struct gt_val
{
	__host__ __device__
	bool operator() (const float &x)
	{
		return x > 0.9;
	}
};

template<typename T>
struct greater_than_val : public thrust::unary_function<T, bool> {
  T val;
  greater_than_val(T val_) : val(val_) {}
  inline __host__ __device__
  bool operator()(T x) const {
    return x > val;
  }
};

using thrust::make_zip_iterator;
using thrust::make_tuple;
using thrust::make_counting_iterator;

int main(){
	std::cout << "hello" << std::endl;

	// float * data = new float[N*N];
	thrust::host_vector<int> seq(N*N);
	// thrust::sequence(seq.begin(), seq.end());

	float *d_data;

	gpuErrchk(cudaMalloc(&d_data, N*N*sizeof(float)));
	GPU_fill_rand(d_data, N, N);

	//cudaMemcpy(data, d_data, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	thrust::device_ptr <float> t_data(d_data);
	t_data[10] = 20.;
	t_data[11] = 22.;
	t_data[12] = 23.;
	t_data[13] = 25.;
	t_data[14] = 21.;
	t_data[15] = 2343.;
	t_data[16] = 11.;

	// thrust::copy(t_data, t_data + N*N, d_X.begin());
	// d_X = data;

	// int ans = 0;
	int numb_pts = 1000;

	thrust::device_vector<float> d_results_data(numb_pts);
	thrust::device_vector<int> d_results_indicies(numb_pts);

	thrust::host_vector<float> results_data(numb_pts);
	thrust::host_vector<int> results_indicies(numb_pts);

	thrust::fill(d_results_data.begin(), d_results_data.end(), 0);
	thrust::fill(d_results_indicies.begin(), d_results_indicies.end(), 0);

	float thresh = 0.9999;

	cudaStream_t streams[10];
	for (int i = 0; i < 10; i++){
		cudaStreamCreate(&(streams[i]));
	}

	int ans;

	for (int i = 0; i < 10; i++){

		// thrust::copy(d_X.begin(),d_X.end(), d_XX.begin());
		// thrust::copy(d_Y.begin(),d_Y.end(), d_YY.begin());
		// thrust::sort_by_key(t_data, t_data + N*N, d_XX.begin());

		// int ans = thrust::reduce(t_data, t_data + N*N, 0, thrust::maximum<float>());

		// ans = thrust::count_if(t_data, t_data+N*N, gt_val());
		// ans = thrust::count_if(d_X.begin(), d_X.end(), gt_val());

		ans = thrust::copy_if(thrust::cuda::par.on(streams[i]),
						make_zip_iterator(make_tuple(t_data, make_counting_iterator(0u))),
						make_zip_iterator(make_tuple(t_data, make_counting_iterator(0u))) + N*N,
						t_data,
						make_zip_iterator(make_tuple(d_results_data.begin(), d_results_indicies.begin())),
						greater_than_val<float>(thresh)
						) - make_zip_iterator(make_tuple(d_results_data.begin(), d_results_indicies.begin()));

		std::cout << "number: " << ans  << std::endl;

		// thrust::copy_if(make_zip_iterator(make_tuple(t_data, seq)),
		// 				make_zip_iterator(make_tuple(t_data, seq)) + N*N,
		// 				t_data,
		// 				make_zip_iterator(make_tuple(d_results_data.begin(), d_results_indicies.begin())),
		// 				greater_than_val<float>(thresh)
		// 				);





	}

	results_data = d_results_data;
	results_indicies = d_results_indicies;
	// std::cout << "ans: " << results_indicies << std::endl;
	// std::cout << "iteration numb: " << i << std::endl;
	for(int i = ans - 4 ; i < min(numb_pts, ans + 4); i++){
		printf("indic[%d, %d] = %f\n", i, results_indicies[i], results_data[i]);
		// std::cout << "indicies[" << i << "] = " << results_indicies[i] << std::endl;
		// std::cout << "data[" << i << "] = " << results_data[i] << std::endl;
	}

	

	cudaFree(d_data);
	// delete[] data;

	return 0;
}





