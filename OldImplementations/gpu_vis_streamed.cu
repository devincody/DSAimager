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

// nvcc -o gpu_vis_streamed gpu_vis_streamed.cu -lsfml-graphics -lcfitsio -O2 -lcufft -std=c++11 -Wno-deprecated-gpu-targets
#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 1024
#define N_ANTENNAS 10
#define N_BASELINES (N_ANTENNAS - 1)*(N_ANTENNAS)/2
#define N_STREAMS 5
#define N_TIMESTEPS 300
#define N_DUMPPOINTS 10000


#include "gpu_vis_streamed.cuh"




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
	// 		       { 1.83569971e+02,  -4.37387873e-05,   1.57832354e-05},
	// 		       {-1.03180029e+02,  -1.03154892e-05,  -2.81970909e-05},
	// 		       {-1.76180029e+02,  -1.63615256e-06,  -3.96178686e-05},
	// 			// 		       { 1.90319971e+02,  -4.48715200e-05,   1.72737512e-05},
       // {-2.03330030e+02,   1.63615256e-06,  -4.39237587e-05},
	// 		       {-9.91080054e+02,  -2.16759905e+02,  -4.00017642e+00}};
	// int ant_order[] = {1, 4, 5, 8, 6, 9, 2, 10, 3, 7};
	float pos_extent =  1299.54002797;

	// std::cout << "Hello, position: " << pos[3][2] << std::endl;

	/***************************************************
	Datastructures for moving data
	***************************************************/
	sf::Uint8 *img = new sf::Uint8[SZ*SZ*N_STREAMS*4]; // temporary image
	if (!img){ std::cerr << "img not allocated" << std::endl;}

	sf::Image image[N_STREAMS];						 // SFML image object

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
	cufftComplex *C = new cufftComplex[SZ*SZ*N_STREAMS];
	if (!C){std::cerr << "Matricies not allocated" << std::endl;}

	cufftComplex *d_A, *d_C, *d_A_cal, *d_B_cal;
	float *d_data, *d_norm;
	int n_pol = 2;
	int n_complex = 2;
	int n_floats_per_freq = (N_BASELINES + 10)*n_pol*n_complex; // # floats per frequency


	gpuErrchk(cudaMalloc(&d_A,      SZ*SZ*N_STREAMS*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_C,      SZ*SZ*N_STREAMS*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_norm,   SZ*SZ*N_STREAMS*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_A_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	gpuErrchk(cudaMalloc(&d_B_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	gpuErrchk(cudaMalloc(&d_data,   n_floats_per_freq*n_usable_idx*N_TIMESTEPS*sizeof(float)));

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
	if (!A_cal){ std::cerr << "A_cal not allocated" << std::endl;}
	if (!B_cal){ std::cerr << "B_cal not allocated" << std::endl;}

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

	cufftHandle plan[N_STREAMS];
	cudaStream_t stream[N_STREAMS];
	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamCreate(&(stream[i])));
		gpuFFTchk(cufftPlan2d(&(plan[i]), SZ, SZ, CUFFT_C2C));
	}

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

	gpuErrchk(cudaHostRegister(data, nrows*sizeof(float), cudaHostRegisterPortable));

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

	int idx;

	cufftComplex thresh;
	thresh.x = 90000.0;
	thresh.y = 0;

	thrust::device_vector<cufftComplex> d_value(N_DUMPPOINTS);
	thrust::device_vector<int>   d_index(N_DUMPPOINTS);
	thrust::host_vector<cufftComplex>   h_value(N_DUMPPOINTS);
	thrust::host_vector<int>     h_index(N_DUMPPOINTS); 


	/***************************************************
	Start continuous Loop
	***************************************************/
	int time[N_STREAMS];
	for (int i = 0; i < N_STREAMS; i++){
		time[i] = 220 + i;
		// t_data[i] = &(d_C[SZ*SZ]);
	}

	int EXIT_FLAG = 0;
	while(!EXIT_FLAG){
		for (int s = 0; s < N_STREAMS; s ++){
			gpuErrchk(cudaStreamSynchronize(stream[s]));
			START_TIMER();

			int img_strt = s*SZ*SZ;
		

			std::cout << "time: " << time[s] << ", stream: " << s << std::endl;
			
			// Memcpy minimum data required for gridding.
			gpuErrchk(cudaMemcpyAsync(&(d_data[time[s] % N_TIMESTEPS]), 
								 &(data[IDX(time[s], idx_usable_start, 0)]), 
								 n_floats_per_freq*n_usable_idx*sizeof(float), 
								 cudaMemcpyHostToDevice,
								 stream[s]));
			/*
			TODO:
			copy data into correct block of memory
			Identify which data is needed when
			*/

			// memset(A, 0, SZ*SZ*sizeof(cufftComplex));
			// memset(C, 0, SZ*SZ*sizeof(cufftComplex));
			// memset(norm, 0, SZ*SZ*sizeof(float));

			// reset workspaces to zero
			gpuErrchk(cudaMemsetAsync(&(d_A[img_strt]),    0, SZ*SZ*sizeof(cufftComplex), stream[s]));
			gpuErrchk(cudaMemsetAsync(&(d_norm[img_strt]), 0, SZ*SZ*sizeof(float), stream[s]));

			std::cout << "calling kernel" << std::endl;

			grid<<<1,45,0,stream[s]>>>(&(d_A[img_strt]),  &(d_data[time[s]]), d_A_cal, d_B_cal, 250, 500);


			std::cout << "Done Filling Matrix" << std::endl;

					
			// gpuErrchk(cudaMemcpy(d_A, A, sizeof(cufftComplex)*SZ*SZ, cudaMemcpyHostToDevice));

			gpuFFTchk(cufftExecC2C(plan[s], &(d_A[img_strt]), &(d_C[img_strt]), CUFFT_FORWARD));

			// real<<<20,32, 0, stream[s]>>>(&(d_C[img_strt]), SZ*SZ);

			thrust::device_ptr <cufftComplex> t_data(&(d_C[img_strt]));

			int ans = thrust::copy_if(thrust::cuda::par.on(stream[s]),
						make_zip_iterator(make_tuple(t_data, make_counting_iterator(0u))),
						make_zip_iterator(make_tuple(t_data, make_counting_iterator(0u))) + SZ*SZ,
						t_data,
						make_zip_iterator(make_tuple(d_value.begin(), d_index.begin())),
						greater_than_val<cufftComplex>(thresh)
						) - make_zip_iterator(make_tuple(d_value.begin(), d_index.begin()));

			h_value = d_value;
			h_index = d_index;

			gpuErrchk(cudaMemcpyAsync(&(C[img_strt]), &(d_C[img_strt]), sizeof(cufftComplex)*SZ*SZ, cudaMemcpyDeviceToHost, stream[s]));

			std::cout << "number of points = " << ans << std::endl;
			float maxx = 0;
			int indexx = 0;
			for (int i = 0; i < ans; i++){
				if (h_value[i].x > maxx){
					// std::cout << "h: " << h_value[i].x;
					maxx = h_value[i].x;
					indexx = h_index[i];
				}
			}

			std::cout << "max value by copy_if = " << maxx << ", at location = " << indexx << std::endl;


			/*
			TODO:
			Do copy_if
			Only copy the copyif vector, also overflow signal for dumping all the data
			send data for comparison

			Calculate coincidences (std::map)
			At end, search through list of all arrays, if proximate data points, include in sum.

			*/

			
			
			/*
			Calculate Statistics
			*/
			float max = 0;
			float avg = 0;
			float mag = 0;
			for (int i = 0; i < SZ*SZ; i++){
				mag = C[i+img_strt].x;//std::abs(C[i+img_strt].x);//*C[i].x + C[i].y*C[i].y);
				if (mag > max){
					max = mag;
				}
				avg += mag;
			}
			avg /= SZ*SZ;

			float stdr = 0;
			for (int i = 0; i < SZ*SZ; i++){
				mag = C[i+img_strt].x;//std::sqrt(C[i+img_strt].x*C[i+img_strt].x);// + C[i+img_strt].y*C[i+img_strt].y);
				stdr += (avg - mag)*(avg - mag);
			}

			stdr = std::sqrt(stdr/(SZ*SZ-1));
			std::cout << "time: " << time[s] << ", max: " << max << ", avg: " << avg << ", std: " << stdr << std::endl;
			

			std::cout << "Writing to disk"<< std::endl;

			float cmax = 2000000.0;
			float cmin = 0;
			float abs = 0;





			// write to image, flip spectra
			for (int i = 0; i < SZ*SZ; i++){
				abs = std::sqrt(C[i+img_strt].x*C[i+img_strt].x);// + C[i+img_strt].y*C[i+img_strt].y);

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

				img[4*idx   + 4*img_strt] = px_color.red;
				img[4*idx+1 + 4*img_strt] = px_color.green;
				img[4*idx+2 + 4*img_strt] = px_color.blue;
				img[4*idx+3 + 4*img_strt] = 255;	
				// std::cout << "img: "<< temp << std::endl;
			}

			image[s].create(SZ, SZ, &(img[SZ*SZ*4*s]));
			
			//Create and save image
			oss.str("");
			oss << std::setfill('0') << std::to_string(time[s]) << std::setw(6);
			title = "cppFrames/" + oss.str() + ".jpg";
			std::cout << "title: " << title << std::endl;
			image[s].saveToFile(title);//"Frames6/" + std::to_string(kp) + img_extension);

			STOP_RECORD_TIMER(gpu_time_ms_solve);// avg_gpu_time += gpu_time_ms_solve; avg_gpu_time_cnt += 1;
			std::cout << "Done (Fourier) Transforming in " << gpu_time_ms_solve <<" ms \n" << std::endl;

			
			if (time[s] == 240){
				EXIT_FLAG = 1;
				break;
			}
			time[s] += N_STREAMS;

		}


	}
	// Garbage collection
	delete[] C;
	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_C));
	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_A_cal));
	gpuErrchk(cudaFree(d_B_cal));
	// gpuFFTchk(cufftDestroy(plan));

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamDestroy(stream[i]));
		gpuFFTchk(cufftDestroy(plan[i]));
	}

	// delete[] norm;
	gpuErrchk(cudaHostUnregister(data));
	delete[] data;
	fits_close_file(fptr, &status);
	return 0;
}








