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

// nvcc -o real realtime_imager.cu -lsfml-graphics -lcfitsio -O2 -lcufft -std=c++11 -Wno-deprecated-gpu-targets
#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 1024											//Size of the 2D FFT
#define N_ANTENNAS 10									//Number of Antennas
#define N_BASELINES (N_ANTENNAS - 1)*(N_ANTENNAS)/2		//Number of Baselines (45)
#define N_STREAMS 5										//Number of Streams
#define N_TIMESTEPS 300									//How much data to hold onto
#define N_DUMPPOINTS 10000								//How many points to identify in copy_if


#include "realtime_imager2.cuh"   


int main (){

	/***************************************************
	Physical Constants, offsets
	***************************************************/
	int idx_usable_start = 300;
	int idx_usable_end   = 1800;
	int n_usable_idx = idx_usable_end - idx_usable_start;
	float f_tot_end    = 1.53;
	float pos_extent =  1299.54002797;

	/***************************************************
	Datastructures for moving data
	***************************************************/

	/* Image Datastructures */
	sf::Uint8 *img = new sf::Uint8[SZ*SZ*N_STREAMS*4]; // image data
	if (!img){ std::cerr << "img not allocated" << std::endl;}

	sf::Image image[N_STREAMS];						 // SFML image object
	std::ostringstream oss;
	std::string title;


	/* gpu timers */
	float gpu_time_ms_solve = -1;
	//float avg_gpu_time = 0;
	//float avg_gpu_time_cnt = 0;

	/* Datastructures for Calcuations */
	cufftComplex *C = new cufftComplex[SZ*SZ*N_STREAMS]; //output image on device
	if (!C){std::cerr << "Matricies not allocated" << std::endl;}

	cufftComplex *d_A, *d_C, *d_A_cal, *d_B_cal;
	float *d_data;
	int n_pol = 2;
	int n_complex = 2;
	int n_floats_per_freq = (N_BASELINES + 10)*n_pol*n_complex; // number of floats per frequency (220)


	gpuErrchk(cudaMalloc(&d_A,      SZ*SZ*N_STREAMS*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_C,      SZ*SZ*N_STREAMS*sizeof(cufftComplex)));
	//gpuErrchk(cudaMalloc(&d_norm,   SZ*SZ*N_STREAMS*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_A_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	gpuErrchk(cudaMalloc(&d_B_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	gpuErrchk(cudaMalloc(&d_data,   n_floats_per_freq*n_usable_idx*N_TIMESTEPS*sizeof(float)));


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
	std::cout << "Got corrections" << std::endl;
	delete[] A_cal;
	delete[] B_cal;


	/***************************************************
	Make Plans for FFT for each stream
	***************************************************/



	cufftHandle plan[N_STREAMS];
	cudaStream_t stream[N_STREAMS];
	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamCreate(&(stream[i])));
		gpuFFTchk(cufftPlan2d(&(plan[i]), SZ, SZ, CUFFT_C2C));
	}

	/***************************************************
	Pull .fits file data into memory
	***************************************************/
	// Variables
	fitsfile *fptr;
	int status = 0;
	int ncols = 0;
	long nrows = 0;
	int px_per_res_elem = 3;
	int column_number = 1;
	int null_values;

	// Collect information about the data
	std::cout << "Opening data file" << std::endl;
	fits_open_data(&fptr, "test.fits", READONLY, &status);
	std::cout << "Opened data file" <<std::endl;
	fits_get_num_rows(fptr, &nrows, &status);
	fits_get_num_cols(fptr, &ncols, &status);
	std::cout <<"rows: " << nrows <<" cols: " << ncols << std::endl;
	std::cout << "Frequencies: " << n_usable_idx << std::endl;

	// Allocate Host Memory for incoming data
	float *data = new float[nrows];
	gpuErrchk(cudaHostRegister(data, nrows*sizeof(float), cudaHostRegisterPortable));

	// Read .fits file
	fits_read_col(fptr, TFLOAT, column_number,
				  1, 1, nrows, NULL, 
				  data,
				  &null_values , &status);

	std::cout << "Done Reading" << std::endl;

	/***************************************************
	Calculate Constants 
	***************************************************/
	float c = 299792458.0;
	// float wavelength;
	float mv = ((float) SZ)/(2.0*pos_extent)* c/(f_tot_end*1E9)/px_per_res_elem;
	std::cout << "mv: " << mv <<std::endl;
	std::cout << "pos_extent: " << pos_extent << std::endl;
	std::cout << "other: " << c/(f_tot_end*1E9)/px_per_res_elem << std::endl;

	/***************************************************
	Thrust constructs for finding maximum values
	***************************************************/

	/* Data Structures for Finding max values */
	cufftComplex thresh;
	thresh.x = 90000.0;
	thresh.y = 0;

	// Vectors to hold data for each stream
	thrust::device_vector<cufftComplex> d_value[N_STREAMS]; //holds values above thresh
	thrust::device_vector<int>   		d_index[N_STREAMS]; //holds indicies of values above thresh
	thrust::device_ptr<cufftComplex> 	t_data[N_STREAMS];	//thrust pointer to data 
	
	// Populate the vectors
	for (int i = 0; i < N_STREAMS; i ++){
		d_value[i] = thrust::device_vector<cufftComplex> (N_DUMPPOINTS);
		d_index[i] = thrust::device_vector<int> (N_DUMPPOINTS);
		t_data[i] = thrust::device_ptr<cufftComplex> (&(d_C[SZ*SZ*i]));
	}

	// Allocate host-side memory for values above thresh
	cufftComplex* h_value = new cufftComplex[N_DUMPPOINTS*N_STREAMS];
	int* h_index = new int[N_DUMPPOINTS*N_STREAMS];
	gpuErrchk(cudaHostRegister(h_value, N_DUMPPOINTS*N_STREAMS*sizeof(float), cudaHostRegisterPortable));
	gpuErrchk(cudaHostRegister(h_index, N_DUMPPOINTS*N_STREAMS*sizeof(float), cudaHostRegisterPortable));



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
		START_TIMER();
		for (int s = 0; s < N_STREAMS; s ++){

			int img_strt = s*SZ*SZ;
		
			#if DEBUG
				std::cout << "time: " << time[s] << ", stream: " << s << std::endl;
			#endif

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

			gpuErrchk(cudaMemsetAsync(&(d_A[img_strt]), 0, SZ*SZ*sizeof(cufftComplex), stream[s]));

			#if DEBUG
				std::cout << "calling kernel" << std::endl;
			#endif
		}

		for (int s = 0; s < N_STREAMS; s ++){
			int img_strt = s*SZ*SZ;
			int smem = 0;
			grid<<<45, 250, smem, stream[s]>>>(&(d_A[img_strt]), d_data, time[s] % N_TIMESTEPS, d_A_cal, d_B_cal, 250, 500, 56.5);

			#if DEBUG
				std::cout << "Done Filling Matrix" << std::endl;
			#endif


			/* Execute FFT */
			gpuFFTchk(cufftExecC2C(plan[s], &(d_A[img_strt]), &(d_C[img_strt]), CUFFT_FORWARD));

			/* Find Largest Values */
			int ans = thrust::copy_if(thrust::cuda::par.on(stream[s]),
						make_zip_iterator(make_tuple(t_data[s], make_counting_iterator(0u))),
						make_zip_iterator(make_tuple(t_data[s], make_counting_iterator(0u))) + SZ*SZ,
						t_data[s],
						make_zip_iterator(make_tuple(d_value[s].begin(), d_index[s].begin())),
						greater_than_val<cufftComplex>(thresh)
						) - make_zip_iterator(make_tuple(d_value[s].begin(), d_index[s].begin()));


			/* Copy Largest Values to RAM */
			if (ans < N_DUMPPOINTS && ans > 0){
				gpuErrchk(cudaMemcpyAsync(&(h_value[N_DUMPPOINTS*s]), thrust::raw_pointer_cast(d_value[s].data()), ans*sizeof(cufftComplex), cudaMemcpyDeviceToHost, stream[s]));
				gpuErrchk(cudaMemcpyAsync(&(h_index[N_DUMPPOINTS*s]), thrust::raw_pointer_cast(d_index[s].data()), ans*sizeof(int), cudaMemcpyDeviceToHost, stream[s]));
			}

			// gpuErrchk(cudaMemcpyAsync(&(C[img_strt]), &(d_C[img_strt]), sizeof(cufftComplex)*SZ*SZ, cudaMemcpyDeviceToHost, stream[s]));
			
			gpuErrchk(cudaStreamSynchronize(stream[s]));
			#if DEBUG
				std::cout << "number of points = " << ans << std::endl;
			#endif

			float maxx = 0;
			int indexx = 0;
			for (int i = 0; i < ans; i++){
				if (h_value[i + N_DUMPPOINTS*s].x > maxx){
					// std::cout << "h: " << h_value[i].x;
					maxx = h_value[i + N_DUMPPOINTS*s].x;
					indexx = h_index[i + N_DUMPPOINTS*s];
				}
			}
			std::cout << "time: " << time[s] << ", max value by copy_if = " << maxx << ", at location = " << indexx << std::endl;

			// /*
			// TODO:
			// Do copy_if
			// Only copy the copyif vector, also overflow signal for dumping all the data
			// send data for comparison

			// Calculate coincidences (std::map)
			// At end, search through list of all arrays, if proximate data points, include in sum.

			// */

			#if DEBUG
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


				int idx;


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
			#endif
			
			if (time[s] == 240){
				EXIT_FLAG = 1;
				break;
			}
			time[s] += N_STREAMS;

		}
		STOP_RECORD_TIMER(gpu_time_ms_solve);// avg_gpu_time += gpu_time_ms_solve; avg_gpu_time_cnt += 1;
		std::cout << "Done (Fourier) Transforming in " << gpu_time_ms_solve <<" ms \n" << std::endl;

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
	gpuErrchk(cudaHostUnregister(h_value));
	gpuErrchk(cudaHostUnregister(h_index));
	delete[] data;
	delete[] h_value;
	delete[] h_index;
	fits_close_file(fptr, &status);
	return 0;
}
































