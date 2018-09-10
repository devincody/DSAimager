#include <iostream>
#include <complex>
#include <sstream>
#include <iomanip>
#include <string>

#include <fftw3.h>
#include <SFML/Graphics.hpp>

#include "fitsio.h"
#include "bitmap_image.hpp"

//g++ vis.cpp -o vis -lfftw3 -lsfml-graphics -lcfitsio
//g++ vis.cpp -o vis -lfftw3 -lsfml-graphics -lcfitsio -std=c++14 -O2
#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 8192

using namespace std::complex_literals;

int main (){

	// DSA constants
	int f_index_start = 200;
	int f_index_end   = 1848;
	int f_samples = f_index_end - f_index_start;
	int tot_f_index = 2048;

	double f_freq_start  = 1.28;
	double f_freq_end    = 1.53;
	double bw = f_freq_end - f_freq_start;
	int n_antennas = 10;
	double pos[10][3] = {{  2.32700165e-05,   4.43329978e+02,   1.02765904e-05},
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
	double pos_extent =  1299.54002797;
	std::complex<double> cx_zero = 0i;
	std::complex<double> cx_i = 1i;

	std::cout << "hello: " << pos[3][2] << std::endl;

	// Workspace data structures
	sf::Uint8 *img = new sf::Uint8[SZ*SZ*4]; //temporary image
	sf::Image image;						//final image object

	fftw_plan plan;
	fftw_complex *A, *C;
	A = (fftw_complex*) fftw_malloc(SZ*SZ*sizeof(fftw_complex));
	C = (fftw_complex*) fftw_malloc(SZ*SZ*sizeof(fftw_complex));
	// arma::cx_mat A(SZ, SZ); // complex mat of doubles for visibility space
	// arma::cx_mat C(SZ, SZ); // for Image

	
	plan = fftw_plan_dft_2d(SZ, SZ, A, C, FFTW_FORWARD, FFTW_ESTIMATE);

	double* norm = new double[SZ*SZ];
	// arma::mat norm(SZ, SZ);

	for (int i = 0; i < SZ*SZ; i++){
		A[i][0] = 0; A[i][1] = 0;
		C[i][0] = 0; C[i][0] = 0;
	}
	// A.fill(0);
	// norm.fill(0);

	// Variables for FITS file
	fitsfile *fptr;
	char card[FLEN_CARD];
	int status = 0;
	int ncols = 0;
	long nrows = 0;
	int px_per_res_elem = 3;

	fits_open_data(&fptr, "../test.fits", READONLY, &status);
	fits_get_num_rows(fptr, &nrows, &status);
	fits_get_num_cols(fptr, &ncols, &status);
	std::cout <<"rows: " << nrows <<" cols: " << ncols << std::endl;
	std::cout << "Frequencies: " << f_samples << std::endl;

	int column_number = 1;
	int null_values;
	double *data = new double[nrows];
	fits_read_col(fptr, TDOUBLE, column_number,
				  1, 1, nrows, NULL, 
				  data,
				  &null_values , &status);

	std::cout << "Done Reading" << std::endl;

	double c = 299792458.0;
	double wavelength;
	double mv = ((double) SZ)/(2.0*pos_extent)* c/(f_freq_end*1E9)/px_per_res_elem;
	std::cout << "mv: " << mv <<std::endl;
	std::cout << "pos_extent: " << pos_extent << std::endl;
	std::cout << "other: " << c/(f_freq_end*1E9)/px_per_res_elem << std::endl;

	int x, y, sm, lg, idx;

	double cor[2] = {0,0};
	// std::complex<double> cor;

	// for (int i = 0; i < 220; i++){
	// 	std::cout << "data[231,203,"<< i <<"] = " << data[IDX(231,203,i)] <<std::endl;
	// }

	double second_pol = 1;

	for (int time = 231; time < 232; time ++){
		std::cout << "time: " << time << std::endl;
		for (int i = 0; i < n_antennas; i++){
			for (int j = 0; j < n_antennas; j++){
				if (i != j){
					for (int f = f_index_start; f < f_index_end; f++){
						wavelength = c/(1E9*(f_freq_end - static_cast<double>(f)/tot_f_index*bw ));

						x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength));
						y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength));
						

						if (i < j){
							sm = i; lg = j;
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;
							cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
							cor[1] = -data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
						} else {
							sm = j; lg = i;
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;
							cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
							cor[1] =  data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
						}


						// A(x,y) += cor;
						// norm(x,y) += std::abs(cor);
						A[x*SZ+y][0] += cor[0];
						A[x*SZ+y][1] += cor[1];

						norm[x*SZ+y] += sqrt(cor[0]*cor[0] + cor[1]*cor[1]);

					}
				}
			}
		}
	}

	std::cout << "Done Filling Matrix" << std::endl;

	for (int i = 0; i < SZ*SZ; i++){
		if (norm[i] > 0){
			A[i][0] /= norm[i];
			A[i][1] /= norm[i];
		}
	}
	// for (int i = 0; i < SZ; i++){
	// 	for (int j = 0; j < SZ; j++){
	// 		if (norm(i,j) > 0){
	// 			// std::cout << "A("<< i << ", " << j << ") = " << A(i,j) << std::endl;
	// 			A(i,j) /= norm(i,j);
	// 			// std::cout << "A("<< i << ", " << j << ") = " << A(i,j) << std::endl;
	// 		}
	// 	}
	// }

	std::cout << "Done Normalizing" << std::endl;

	for (int i = 0; i < 10; i++)
		fftw_execute(plan);
	// C = fft2(A);
	std::cout << "Done (Fourier) Transforming" << std::endl;


	std::cout << "Writing"<< std::endl;

	double cmax = 80000;
	double cmin = 0;
	double abs = 0;

	for (int i = 0; i < SZ*SZ; i++){
		abs = C[i][0]*C[i][0] + C[i][1]*C[i][1];
		
		// if (abs > 60000)
		// 	std::cout << "abs: " << abs << std::endl;

		unsigned int temp = static_cast<unsigned int> (999*((abs-cmin)/(cmax-cmin)));

		rgb_t px_color = jet_colormap[temp];


		// idx = i; //remap based on flipped array
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
	// arma::mat D = square(abs(C));

	// std::cout << "min: " << cmin << " max: " << cmax << std::endl;

	// f_data << "K2 = " << k2 << std::endl;
	// for (int i = 0; i < SZ; i++){

	// 	for (int j = 0; j < SZ; j++){
	// 		if (i < SZ/2 && j < SZ/2){
	// 			x = i + SZ/2; y = j + SZ/2;
	// 		} else if (i >= SZ/2 && j < SZ/2){
	// 			x = i - SZ/2; y = j + SZ/2;
	// 		} else if (i >= SZ/2 && j >= SZ/2){
	// 			x = i - SZ/2; y = j - SZ/2;
	// 		} else { //i < SZ/2 ** j > SZ/2
	// 			x = i + SZ/2; y = j - SZ/2;
	// 		}

	// 		// if (x == SZ || y == SZ){
	// 		// std::cout << "Error: (" << x <<","<<y<<")" << std::endl;
	// 		// std::cout << "Error: (" << i <<","<<j<<")" << std::endl;
	// 		// }
			


	// 		unsigned int temp = static_cast<unsigned int> (999*((D(x,y)-cmin)/(cmax-cmin)));

	// 		rgb_t px_color = jet_colormap[temp];
	// 		idx = (i*SZ+j);
	// 		img[4*idx] = px_color.red;
	// 		img[4*idx+1] = px_color.green;
	// 		img[4*idx+2] = px_color.blue;
	// 		img[4*idx+3] = 255;

	// 	}
	// }

	image.create(SZ, SZ, img); //create image from Uint8 array
	std::ostringstream oss;
	oss << std::setfill('0') << std::to_string(0) << std::setw(4);
	std::string title = "cppFrames/" + oss.str() + ".jpg";
	image.saveToFile(title);//"Frames6/" + std::to_string(kp) + img_extension);




	fftw_free(A);
	fftw_free(C);
	fftw_destroy_plan(plan);

	delete[] norm;
	delete[] data;
	fits_close_file(fptr, &status);
	return 0;
}



