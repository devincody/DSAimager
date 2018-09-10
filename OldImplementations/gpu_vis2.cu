#include <iostream>
#include <complex>
#include <sstream>
#include <iomanip>
#include <string>
#include <stdio.h> // printf for device functions

#include <SFML/Graphics.hpp>

#include "fitsio.h" // fits files for fake data
#include "bitmap_image.hpp" // colorscheme

#include <cufft.h>
#include <curand.h>
#include <cuda_runtime.h>

// nvcc -o gpuvis2 gpu_vis2.cu -lsfml-graphics -lcfitsio -O2 -lcufft -std=c++11 -Wno-deprecated-gpu-targets
#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 1048
#define N_ANTENNAS 10
#define N_BASELINES (N_ANTENNAS - 1)*(N_ANTENNAS)/2



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


int main (){

	/***************************************************
	Constants of operation
	***************************************************/
	int idx_usable_start = 300;
	int idx_usable_end   = 1800;
	int n_usable_idx = idx_usable_end - idx_usable_start;
	// int tot_f_index = 2048;

	// float f_total_start  = 1.28;
	float f_tot_end    = 1.53;
	// float bw = f_tot_end - f_total_start;
	// int n_antennas = 10;
	// float pos[10][3] = {{  2.32700165e-05,   4.43329978e+02,   1.02765904e-05},
	// 		       { 2.21098953e-05,   4.36579978e+02,   9.64180626e-06},
	// 		       { 1.14039406e-05,   3.42179978e+02,   4.94604756e-06},
	// 		       {-1.14039406e-05,   1.53679978e+02,  -4.94604756e-06},
	// 		       { 1.90319971e+02,  -4.48715200e-05,   1.72737512e-05},
	// 		       { 1.83569971e+02,  -4.37387873e-05,   1.57832354e-05},
	// 		       {-1.03180029e+02,  -1.03154892e-05,  -2.81970909e-05},
	// 		       {-1.76180029e+02,  -1.63615256e-06,  -3.96178686e-05},
	// 		       {-2.03330030e+02,   1.63615256e-06,  -4.39237587e-05},
	// 		       {-9.91080054e+02,  -2.16759905e+02,  -4.00017642e+00}};
	// int ant_order[] = {1, 4, 5, 8, 6, 9, 2, 10, 3, 7};
	float pos_extent =  1299.54002797;

	// std::cout << "Hello, position: " << pos[3][2] << std::endl;

	/***************************************************
	Datastructures for moving data
	***************************************************/
	sf::Uint8 *img = new sf::Uint8[SZ*SZ*4]; // temporary image
	sf::Image image;						 // SFML image object
	image.create(SZ, SZ, img); //create image from Uint8 array
	std::ostringstream oss;
	std::string title;


	// gpu timers
	float gpu_time_ms_solve = -1;
	float avg_gpu_time = 0;
	float avg_gpu_time_cnt = 0;


	// FFT plans and workspaces
	// fftw_plan plan;
	// fftw_complex *A, *C;
	// A = (fftw_complex*) fftw_malloc(SZ*SZ*sizeof(fftw_complex));
	// C = (fftw_complex*) fftw_malloc(SZ*SZ*sizeof(fftw_complex));

	// cufftComplex *A = (cufftComplex*) malloc(sizeof(cufftComplex)*SZ*SZ);
	// cufftComplex *C = (cufftComplex*) malloc(sizeof(cufftComplex)*SZ*SZ);

	// cufftComplex *A = new cufftComplex[SZ*SZ]; //(sizeof(cufftComplex)*SZ*SZ);
	cufftComplex *C = new cufftComplex[SZ*SZ];

	if (!C){
		std::cerr << "Matricies not allocated" << std::endl;
	}

	cufftComplex *d_A, *d_C, *d_A_cal, *d_B_cal;
	float *d_data, *d_norm;
	int n_pol = 2;
	int n_complex = 2;
	int n_floats_per_freq = (N_BASELINES + 10)*n_pol*n_complex; // # floats per frequency


	gpuErrchk(cudaMalloc(&d_A,      SZ*SZ*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_C,      SZ*SZ*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_norm,   SZ*SZ*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_A_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	gpuErrchk(cudaMalloc(&d_B_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	gpuErrchk(cudaMalloc(&d_data,   n_floats_per_freq*n_usable_idx*sizeof(float)));

	// int n_baselines = 55;

	// gpuErrchk(cudaMalloc( &d_data, n_floats_per_freq*n_usable_idx*sizeof(float)));
	//d_data is indexed by [frequency, baseline]


	/***************************************************
	Get visibility corrections and Upload to GPU
	***************************************************/
	// A_cal is indexed by [baseline, frequency]

	std::ifstream A_real, A_imag, B_real, B_imag;
	A_real.open("AA_real.txt");
	A_imag.open("AA_imag.txt");
	B_real.open("BB_real.txt");
	B_imag.open("BB_imag.txt");

	cufftComplex* A_cal = new cufftComplex[N_BASELINES*n_usable_idx];
	cufftComplex* B_cal = new cufftComplex[N_BASELINES*n_usable_idx];

	for (int i = 0; i < N_BASELINES; i++){
		for (int j = 0; j < n_usable_idx; j++){
			A_real >> A_cal[i*n_usable_idx + j].x;
			A_imag >> A_cal[i*n_usable_idx + j].y;
			B_real >> B_cal[i*n_usable_idx + j].x;
			B_imag >> B_cal[i*n_usable_idx + j].y;
		}
	}

	gpuErrchk(cudaMemcpy(d_A_cal, A_cal, N_BASELINES*n_usable_idx*sizeof(cufftComplex), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B_cal, B_cal, N_BASELINES*n_usable_idx*sizeof(cufftComplex), cudaMemcpyHostToDevice));

	delete[] A_cal;
	delete[] B_cal;


	/***************************************************
	Make Plan for FFT
	***************************************************/

	std::cout << "Got corrections" << std::endl;

	cufftHandle plan;
	gpuFFTchk(cufftPlan2d(&plan, SZ, SZ, CUFFT_C2C));
	
	// plan = fftw_plan_dft_2d(SZ, SZ, A, C, FFTW_FORWARD, FFTW_ESTIMATE);
	std::cout << "Made plan" << std::endl;

	// float* norm = new float[SZ*SZ];



	/***************************************************
	Pull fits data into memory
	***************************************************/

	fitsfile *fptr;
	int status = 0;
	int ncols = 0;
	long nrows = 0;
	int px_per_res_elem = 3;

	std::cout << "Opening data file" << std::endl;
	fits_open_data(&fptr, "test.fits", READONLY, &status);
	std::cout << "Opened data file" <<std::endl;
	fits_get_num_rows(fptr, &nrows, &status);
	fits_get_num_cols(fptr, &ncols, &status);
	std::cout <<"rows: " << nrows <<" cols: " << ncols << std::endl;
	std::cout << "Frequencies: " << n_usable_idx << std::endl;

	int column_number = 1;
	int null_values;
	float *data = new float[nrows];
	fits_read_col(fptr, TFLOAT, column_number,
				  1, 1, nrows, NULL, 
				  data,
				  &null_values , &status);

	std::cout << "Done Reading" << std::endl;


    // for (int i = 0; i < 10; i++){
    //         std::cout << "data[" << i << "] = "<< data[IDX(231,300,i)] << std::endl;
    // }

    // gpuErrchk(cudaMemcpy(d_data, &(data[IDX(231, 300, 0)]), 220*1500*sizeof(float), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(&(data[IDX(231, 300, 0)]), d_data, 220*1500*sizeof(float), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 10; i ++){
    //         std::cout << "data[" << i << "] = " << data[IDX(231,300,i)] << std::endl;
    // }

	/***************************************************
	Calculate Constants 
	***************************************************/


	float c = 299792458.0;
	// float wavelength;
	float mv = ((float) SZ)/(2.0*pos_extent)* c/(f_tot_end*1E9)/px_per_res_elem;
	std::cout << "mv: " << mv <<std::endl;
	std::cout << "pos_extent: " << pos_extent << std::endl;
	std::cout << "other: " << c/(f_tot_end*1E9)/px_per_res_elem << std::endl;

	// int x, y, sm, lg;
	int idx;

	// float cor[2] = {0,0};
	// std::complex<float> cor;

	// for (int i = 0; i < 220; i++){
	// 	std::cout << "data[231,203,"<< i <<"] = " << data[IDX(231,203,i)] <<std::endl;
	// }

	// cufftComplex tempa, tempb;

	/***************************************************
	Start continuous Loop
	***************************************************/

	for (int time = 220; time < 240; time ++){
		std::cout << "time: " << time << std::endl;
		
		// Memcpy minimum data required for gridding.
		gpuErrchk(cudaMemcpy(d_data, 
							 &(data[IDX(time, idx_usable_start, 0)]), 
							 n_floats_per_freq*n_usable_idx*sizeof(float), 
							 cudaMemcpyHostToDevice));

		// memset(A, 0, SZ*SZ*sizeof(cufftComplex));
		// memset(C, 0, SZ*SZ*sizeof(cufftComplex));
		// memset(norm, 0, SZ*SZ*sizeof(float));

		// reset workspaces to zero
		gpuErrchk(cudaMemset(d_A, 0, SZ*SZ*sizeof(cufftComplex)));
		gpuErrchk(cudaMemset(d_norm, 0, SZ*SZ*sizeof(float)));

		std::cout << "calling kernel" << std::endl;

		grid<<<1,45>>>(d_A,  d_data, d_A_cal, d_B_cal, 250, 500);
		// d_A is size SZ*SZ*sizeof(cufftComplex)
		// d_data is size 220*1500*sizeof(float)
		// d_X_cal is size 45*1500*sizeof(cufftComplex)
		
		// for (int i = 0; i < n_antennas; i++){
		// 	// std::cout << "ant: " << i << std::endl;
		// 	for (int j = 0; j < n_antennas; j++){
		// 		if (i != j){
		// 			for (int f = idx_usable_start; f < idx_usable_start+idx_usable_end; f++){
		// 				wavelength = c/(1E9*(f_tot_end - static_cast<float>(f)/tot_f_index*bw ));

		// 				x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength));
		// 				y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength));
						

		// 				if (i < j){
		// 					sm = i; lg = j;
		// 					idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;

		// 					// load complex number into temporary variable
		// 					tempa.x = data[IDX(time, f, 4*idx)];
		// 					tempa.y = data[IDX(time, f, 4*idx+1)];

		// 					tempb.x = data[IDX(time, f, 4*idx+2)];
		// 					tempb.y = data[IDX(time, f, 4*idx+3)];

		// 					//std::cout << "tempa: " << tempa << ", a_cal: " << A_cal[idx*1500 + f] << std::endl;

		// 					cx_mult(tempa, A_cal[idx*n_usable_idx + f - idx_usable_start]);
		// 					cx_mult(tempb, B_cal[idx*n_usable_idx + f - idx_usable_start]);

		// 					cor[0] =  tempa.x + tempb.x;
		// 					cor[1] = -tempa.y - tempb.y;
		// 					// deal with each polarization separately then add the real and imag parts to cor
		// 					// cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
		// 					// cor[1] = -data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
		// 				} else {
		// 					sm = j; lg = i;
		// 					idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;

		// 					tempa.x = data[IDX(time, f, 4*idx)];
		// 					tempa.y = data[IDX(time, f, 4*idx+1)];

		// 					tempb.x = data[IDX(time, f, 4*idx+2)];
		// 					tempb.y = data[IDX(time, f, 4*idx+3)];

		// 					//std::cout << "tempa: " << tempa << ", a_cal: " << A_cal[idx*1500 + f][0] << std::endl;

		// 					cx_mult(tempa, A_cal[idx*n_usable_idx + f - idx_usable_start]);
		// 					cx_mult(tempb, B_cal[idx*n_usable_idx + f - idx_usable_start]);

		// 					cor[0] = tempa.x + tempb.x;
		// 					cor[1] = tempa.y + tempb.y;
		// 					// cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
		// 					// cor[1] =  data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
		// 				}


		// 				// A(x,y) += cor;
		// 				// norm(x,y) += std::abs(cor);
		// 				A[x*SZ+y].x += cor[0];
		// 				A[x*SZ+y].y += cor[1];

		// 				norm[x*SZ+y] += sqrt(cor[0]*cor[0] + cor[1]*cor[1]);

		// 			}
		// 		}
		// 	}
		// }


		



		std::cout << "Done Filling Matrix" << std::endl;

				
		// gpuErrchk(cudaMemcpy(d_A, A, sizeof(cufftComplex)*SZ*SZ, cudaMemcpyHostToDevice));

		gpuFFTchk(cufftExecC2C(plan, d_A, d_C, CUFFT_FORWARD));

		gpuErrchk(cudaMemcpy(C, d_C, sizeof(cufftComplex)*SZ*SZ, cudaMemcpyDeviceToHost));


		
		
		/*
		Calculate Statistics
		*/
		float max = 0;
		float avg = 0;
		float mag = 0;
		for (int i = 0; i < SZ*SZ; i++){
			mag = std::abs(C[i].x);//*C[i].x + C[i].y*C[i].y);
			if (mag > max){
				max = mag;
			}
			avg += mag;
		}
		avg /= SZ*SZ;

		float stdr = 0;
		for (int i = 0; i < SZ*SZ; i++){
			mag = std::sqrt(C[i].x*C[i].x + C[i].y*C[i].y);
			stdr += (avg - mag)*(avg - mag);
		}

		stdr = std::sqrt(stdr/(SZ*SZ-1));
		std::cout << "time: " << time << ", max: " << max << ", avg: " << avg << ", std: " << stdr << std::endl;
		

		std::cout << "Writing to disk"<< std::endl;

		float cmax = 2000000.0/3.0;
		float cmin = 0;
		float abs = 0;





		// write to image, flip spectra
		for (int i = 0; i < SZ*SZ; i++){
			abs = std::sqrt(C[i].x*C[i].x + C[i].y*C[i].y);

			unsigned int temp = static_cast<unsigned int> (999*((abs-cmin)/(cmax-cmin)));

			if (temp > 999){
				temp = 999;
			}

			rgb_t px_color = jet_colormap[temp];


			//remap based on flipped array
			if(i < SZ*SZ/2){
				//first half of array, adding
				if((i%SZ) < SZ/2){
					//up left quadrent
					idx = i + SZ*SZ/2 + SZ/2;
				} else {
					//up right quadrent
					idx = i + SZ*SZ/2 - SZ/2;
				}
			} else{
				if((i%SZ) < SZ/2){
					//dwn left quadrent
					idx = i - (SZ*SZ/2 - SZ/2);

				} else {
					//dwm right quadrent
					idx = i - (SZ*SZ/2 + SZ/2);
				}
			}

			img[4*idx] = px_color.red;
			img[4*idx+1] = px_color.green;
			img[4*idx+2] = px_color.blue;
			img[4*idx+3] = 255;	
		}


		START_TIMER();
		//Create and save image
		oss.str("");
		oss << std::setfill('0') << std::to_string(time) << std::setw(4);
		title = "cppFrames/" + oss.str() + ".jpg";
		image.saveToFile(title);//"Frames6/" + std::to_string(kp) + img_extension);

		STOP_RECORD_TIMER(gpu_time_ms_solve); avg_gpu_time += gpu_time_ms_solve; avg_gpu_time_cnt += 1;
		std::cout << "Done (Fourier) Transforming in " << gpu_time_ms_solve <<" ms " << std::endl;


	}

	std::cout << "Average solve time: " << avg_gpu_time/avg_gpu_time_cnt << " ms " << std::endl;

	// Garbage collection
	delete[] C;
	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_C));
	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_A_cal));
	gpuErrchk(cudaFree(d_B_cal));
	gpuFFTchk(cufftDestroy(plan));

	// delete[] norm;
	delete[] data;
	fits_close_file(fptr, &status);
	return 0;
}








