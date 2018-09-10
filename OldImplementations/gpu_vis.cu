#include <iostream>
#include <complex>
#include <sstream>
#include <iomanip>
#include <string>

#include <SFML/Graphics.hpp>

#include "fitsio.h"
#include "bitmap_image.hpp"

#include <cufft.h>
#include <curand.h>
#include <cuda_runtime.h>

// nvcc -o gpuvis gpu_vis.cu -lsfml-graphics -lcfitsio -O2 -lcufft -std=c++11 -Wno-deprecated-gpu-targets
#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 2048

// using namespace std::complex_literals;


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




void cx_mult (cufftComplex &a, cufftComplex b);


int main (){

	// DSA constants
	int f_index_start = 300;
	int f_index_end   = 1800;
	int f_samples = f_index_end - f_index_start;
	int tot_f_index = 2048;

	float f_freq_start  = 1.28;
	float f_freq_end    = 1.53;
	float bw = f_freq_end - f_freq_start;
	int n_antennas = 10;
	float pos[10][3] = {{  2.32700165e-05,   4.43329978e+02,   1.02765904e-05},
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

	std::cout << "Hello, position: " << pos[3][2] << std::endl;

	// Workspace data structures
	sf::Uint8 *img = new sf::Uint8[SZ*SZ*4]; // temporary image
	sf::Image image;						 // SFML image object

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

	cufftComplex *A = new cufftComplex[SZ*SZ]; //(sizeof(cufftComplex)*SZ*SZ);
	cufftComplex *C = new cufftComplex[SZ*SZ];

	if (!A || !C){
		std::cerr << "Matricies not allocated" << std::endl;
	}

	cufftComplex *d_A, *d_C;

	gpuErrchk(cudaMalloc(&d_A, SZ*SZ*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_C, SZ*SZ*sizeof(cufftComplex)));


	// Get visibility corrections
	std::ifstream A_real, A_imag, B_real, B_imag;
	A_real.open("AA_real.txt");
	A_imag.open("AA_imag.txt");
	B_real.open("BB_real.txt");
	B_imag.open("BB_imag.txt");

	cufftComplex* A_cal = new cufftComplex[45*f_samples];
	cufftComplex* B_cal = new cufftComplex[45*f_samples];

	for (int i = 0; i < 45; i++){
		for (int j = 0; j < f_samples; j++){
			A_real >> A_cal[i*f_samples + j].x;
			A_imag >> A_cal[i*f_samples + j].y;
			B_real >> B_cal[i*f_samples + j].x;
			B_imag >> B_cal[i*f_samples + j].y;
		}
	}

	std::cout << "Got corrections" << std::endl;

	cufftHandle plan;
	gpuFFTchk(cufftPlan2d(&plan, SZ, SZ, CUFFT_C2C));
	
	// plan = fftw_plan_dft_2d(SZ, SZ, A, C, FFTW_FORWARD, FFTW_ESTIMATE);
	std::cout << "Made plan" << std::endl;

	float* norm = new float[SZ*SZ];



	// Variables for FITS file
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
	std::cout << "Frequencies: " << f_samples << std::endl;

	int column_number = 1;
	int null_values;
	float *data = new float[nrows];
	fits_read_col(fptr, TFLOAT, column_number,
				  1, 1, nrows, NULL, 
				  data,
				  &null_values , &status);
	std::cout << "Done Reading" << std::endl;

	float c = 299792458.0;
	float wavelength;
	float mv = ((float) SZ)/(2.0*pos_extent)* c/(f_freq_end*1E9)/px_per_res_elem;
	std::cout << "mv: " << mv <<std::endl;
	std::cout << "pos_extent: " << pos_extent << std::endl;
	std::cout << "other: " << c/(f_freq_end*1E9)/px_per_res_elem << std::endl;

	int x, y, sm, lg, idx;

	float cor[2] = {0,0};
	// std::complex<float> cor;

	// for (int i = 0; i < 220; i++){
	// 	std::cout << "data[231,203,"<< i <<"] = " << data[IDX(231,203,i)] <<std::endl;
	// }

	cufftComplex tempa, tempb;

	

	for (int time = 220; time < 240; time ++){
		std::cout << "time: " << time << std::endl;

			
		// // Set memory to zero.
		// for (int i = 0; i < SZ*SZ; i++){
		// 	A[i].x = 0; A[i].y = 0;
		// 	C[i].x = 0; C[i].y = 0;
		// 	norm[i] = 0;
		// }

		memset(A, 0, SZ*SZ*sizeof(cufftComplex));
		memset(C, 0, SZ*SZ*sizeof(cufftComplex));
		memset(norm, 0, SZ*SZ*sizeof(float));


		cufftComplex tempc;

		for (int i = 0; i < n_antennas; i++){
			// std::cout << "ant: " << i << std::endl;
			for (int j = 0; j < n_antennas; j++){
				if (i != j){
					for (int f = f_index_start; f < f_index_end; f++){
						wavelength = c/(1E9*(f_freq_end - static_cast<float>(f)/tot_f_index*bw ));

						x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength));
						y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength));
						

						if (i < j){
							sm = i; lg = j;
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;

							// load complex number into temporary variable
							tempa.x = data[IDX(time, f, 4*idx)];
							tempa.y = data[IDX(time, f, 4*idx+1)];

							tempb.x = data[IDX(time, f, 4*idx+2)];
							tempb.y = data[IDX(time, f, 4*idx+3)];

							//std::cout << "tempa: " << tempa << ", a_cal: " << A_cal[idx*1500 + f] << std::endl;
							tempc.x = tempa.x;
							tempc.y = tempa.y;

							cx_mult(tempa, A_cal[idx*f_samples + f - f_index_start]);
							cx_mult(tempb, B_cal[idx*f_samples + f - f_index_start]);

							// printf("IDX(time, f, 4*idx) = %d => %f + %fj   vs.  %f + %fj\n", IDX(time, f, 4*idx), tempa.x, tempa.y, tempc.x, tempc.y);

							cor[0] =  tempa.x + tempb.x;
							cor[1] = -tempa.y - tempb.y;
							// deal with each polarization separately then add the real and imag parts to cor
							// cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
							// cor[1] = -data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
						} else {
							sm = j; lg = i;
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;

							tempa.x = data[IDX(time, f, 4*idx)];
							tempa.y = data[IDX(time, f, 4*idx+1)];

							tempb.x = data[IDX(time, f, 4*idx+2)];
							tempb.y = data[IDX(time, f, 4*idx+3)];

							//std::cout << "tempa: " << tempa << ", a_cal: " << A_cal[idx*1500 + f][0] << std::endl;

							cx_mult(tempa, A_cal[idx*f_samples + f - f_index_start]);
							cx_mult(tempb, B_cal[idx*f_samples + f - f_index_start]);

							cor[0] = tempa.x + tempb.x;
							cor[1] = tempa.y + tempb.y;
							// cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
							// cor[1] =  data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
						}


						// A(x,y) += cor;
						// norm(x,y) += std::abs(cor);
						float nm = sqrt(cor[0]*cor[0] + cor[1]*cor[1]);
						A[x*SZ+y].x += 1;//cor[0];
						A[x*SZ+y].y += 0;//cor[1];

						norm[x*SZ+y] += nm;

					}
				}
			}
		}

		// int f = 1000;
		// int i = 3;
		// int j = 9;
		// wavelength = c/(1E9*(f_freq_end - static_cast<float>(f)/tot_f_index*bw ));

		// x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength));
		// y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength));

		// std:: cout << "A[" << x <<"." << y<< "] = " << A[x*SZ+y].x  << " + " << A[x*SZ+y].y << "j" << std::endl;

		// x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[j]-1][0]-pos[ant_order[i]-1][0])/wavelength));
		// y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[j]-1][1]-pos[ant_order[i]-1][1])/wavelength));

		// std:: cout << "A[" << x <<"." << y<< "] = " << A[x*SZ+y].x  << " + " << A[x*SZ+y].y << "j" << std::endl;

		// std::cout << "Done Filling Matrix" << std::endl;

		// for (int i = 0; i < SZ*SZ; i++){
		// 	if (norm[i] > 0){
		// 		A[i].x /= norm[i];
		// 		A[i].y /= norm[i];
		// 	}
		// }




		// std::cout << "Done Normalizing" << std::endl;

		START_TIMER();			
		gpuErrchk(cudaMemcpy(d_A, A, sizeof(cufftComplex)*SZ*SZ, cudaMemcpyHostToDevice));

		gpuFFTchk(cufftExecC2C(plan, d_A, d_C, CUFFT_FORWARD));



		gpuErrchk(cudaMemcpy(C, d_C, sizeof(cufftComplex)*SZ*SZ, cudaMemcpyDeviceToHost));
		// fftw_execute(plan);
		
		STOP_RECORD_TIMER(gpu_time_ms_solve);
		avg_gpu_time += gpu_time_ms_solve;
		avg_gpu_time_cnt += 1;

		

		std::cout << "Done (Fourier) Transforming in " << gpu_time_ms_solve <<" ms " << std::endl;


		std::cout << "Writing"<< std::endl;

		float cmax = 4000000;
		float cmin = 0;
		float abs = 0;

		// write to image, flip spectra
		for (int i = 0; i < SZ*SZ; i++){
			// std::cout << "c[" << i << "] = " << C[i].x << " + " << C[i].y << "j" << std::endl;
			abs = C[i].x*C[i].x + C[i].y*C[i].y;

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


		/*
		Calculate Statistics
		*/
		float max = 0;
		float avg = 0;
		float mag = 0;
		for (int i = 0; i < SZ*SZ; i++){
			mag = std::sqrt(C[i].x*C[i].x + C[i].y*C[i].y);
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



		//Create and save image
		image.create(SZ, SZ, img); //create image from Uint8 array
		std::ostringstream oss;
		oss << std::setfill('0') << std::to_string(time) << std::setw(4);
		std::string title = "cppFrames/" + oss.str() + ".jpg";
		image.saveToFile(title);//"Frames6/" + std::to_string(kp) + img_extension);
	}

	std::cout << "Average solve time: " << avg_gpu_time/avg_gpu_time_cnt << " ms " << std::endl;

	// Garbage collection
	delete[] A;
	delete[] C;
	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_C));
	gpuFFTchk(cufftDestroy(plan));

	delete[] norm;
	delete[] data;
	fits_close_file(fptr, &status);
	return 0;
}



void cx_mult (cufftComplex &a, cufftComplex b){
	cufftComplex temp;
	temp.x = a.x*b.x-a.y*b.y;
	temp.y = a.x*b.y + a.y*b.x;
	a.x = temp.x; a.y = temp.y;
}


