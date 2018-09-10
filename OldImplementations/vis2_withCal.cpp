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
//g++ vis2_withCAl.cpp -o viscal -lfftw3 -lsfml-graphics -lcfitsio -std=c++14 -O2
#define IDX(t,f,n) (((t)*220*2048) + ((f)*220) + (n))
#define SZ 2048

using namespace std::complex_literals;

void cx_mult (fftw_complex &a, fftw_complex b);


int main (){

	// DSA constants
	int f_index_start = 300;
	int f_index_end   = 1800;
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
	// std::complex<double> cx_zero = 0i;
	// std::complex<double> cx_i = 1i;

	std::cout << "Hello, position: " << pos[3][2] << std::endl;

	// Workspace data structures
	sf::Uint8 *img = new sf::Uint8[SZ*SZ*4]; // temporary image
	sf::Image image;						 // SFML image object


	// FFT plans and workspaces
	fftw_plan plan;
	fftw_complex *A, *C;
	A = (fftw_complex*) fftw_malloc(SZ*SZ*sizeof(fftw_complex));
	C = (fftw_complex*) fftw_malloc(SZ*SZ*sizeof(fftw_complex));
	// arma::cx_mat A(SZ, SZ); // complex mat of doubles for visibility space
	// arma::cx_mat C(SZ, SZ); // for Image


	// Get visibility corrections
	std::ifstream A_real, A_imag, B_real, B_imag;
	A_real.open("AA_real.txt");
	A_imag.open("AA_imag.txt");
	B_real.open("BB_real.txt");
	B_imag.open("BB_imag.txt");

	fftw_complex* A_cal = new fftw_complex[45*f_samples];
	fftw_complex* B_cal = new fftw_complex[45*f_samples];

	for (int i = 0; i < 45; i++){
		for (int j = 0; j < f_samples; j++){
			A_real >> A_cal[i*f_samples + j][0];
			A_imag >> A_cal[i*f_samples + j][1];
			B_real >> B_cal[i*f_samples + j][0];
			B_imag >> B_cal[i*f_samples + j][1];
		}
	}

	std::cout << "Got corrections" << std::endl;


	
	plan = fftw_plan_dft_2d(SZ, SZ, A, C, FFTW_FORWARD, FFTW_ESTIMATE);
	std::cout << "Made plan" << std::endl;

	double* norm = new double[SZ*SZ];
	// arma::mat norm(SZ, SZ);


	// A.fill(0);
	// norm.fill(0);

	// Variables for FITS file
	fitsfile *fptr;
	char card[FLEN_CARD];
	int status = 0;
	int ncols = 0;
	long nrows = 0;
	int px_per_res_elem = 3;

	std::cout << "Opening data file" << std::endl;
	fits_open_data(&fptr, "../test.fits", READONLY, &status);
	std::cout << "Opened data file" <<std::endl;
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
	fftw_complex tempa, tempb, tempc;

	for (int time = 220; time < 240; time ++){
		std::cout << "time: " << time << std::endl;
		for (int i = 0; i < SZ*SZ; i++){
			A[i][0] = 0; A[i][1] = 0;
			C[i][0] = 0; C[i][0] = 0;
			norm[i] = 0;
		}
		for (int i = 0; i < n_antennas; i++){
			// std::cout << "ant: " << i << std::endl;
			for (int j = 0; j < n_antennas; j++){
				if (i != j){
					for (int f = f_index_start; f < f_index_start+1500; f++){
						wavelength = c/(1E9*(f_freq_end - static_cast<double>(f)/tot_f_index*bw ));

						x = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][0]-pos[ant_order[j]-1][0])/wavelength));
						y = static_cast<int>(SZ/2 + floor(mv*(pos[ant_order[i]-1][1]-pos[ant_order[j]-1][1])/wavelength));
						

						if (i < j){
							sm = i; lg = j;
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;

							// load complex number into temporary variable
							tempa[0] = data[IDX(time, f, 4*idx)];
							tempa[1] = data[IDX(time, f, 4*idx+1)];

							tempb[0] = data[IDX(time, f, 4*idx+2)];
							tempb[1] = data[IDX(time, f, 4*idx+3)];

							//std::cout << "tempa: " << tempa << ", a_cal: " << A_cal[idx*1500 + f] << std::endl;

							cx_mult(tempa, A_cal[idx*f_samples + f - f_index_start]);
							cx_mult(tempb, B_cal[idx*f_samples + f - f_index_start]);

							cor[0] = tempa[0] + tempb[0];
							cor[1] = -tempa[1] - tempb[1];
							// deal with each polarization separately then add the real and imag parts to cor
							// cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
							// cor[1] = -data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
						} else {
							sm = j; lg = i;
							idx = lg-sm-1 + 45 - (9-sm)*(10-sm)/2;

							tempa[0] = data[IDX(time, f, 4*idx)];
							tempa[1] = data[IDX(time, f, 4*idx+1)];

							tempb[0] = data[IDX(time, f, 4*idx+2)];
							tempb[1] = data[IDX(time, f, 4*idx+3)];

							//std::cout << "tempa: " << tempa << ", a_cal: " << A_cal[idx*1500 + f][0] << std::endl;

							tempc[0] = A_cal[idx*f_samples + f - f_index_start][0];
							tempc[1] = A_cal[idx*f_samples + f - f_index_start][1];
							cx_mult(tempa, tempc);
							tempc[0] = B_cal[idx*f_samples + f - f_index_start][0];
							tempc[1] = B_cal[idx*f_samples + f - f_index_start][1];
							cx_mult(tempb, tempc);

							cor[0] = tempa[0] + tempb[0];
							cor[1] =+tempa[1] + tempb[1];
							// cor[0] =  data[IDX(time, f, 4*idx)]   + second_pol*data[IDX(time, f, 4*idx+2)];
							// cor[1] =  data[IDX(time, f, 4*idx+1)] + second_pol*data[IDX(time, f, 4*idx+3)];
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

		std::cout << "Done Filling Matrix" << std::endl;

		for (int i = 0; i < SZ*SZ; i++){
			if (norm[i] > 0){
				A[i][0] /= norm[i];
				A[i][1] /= norm[i];
			}
		}

		std::cout << "Done Normalizing" << std::endl;

		//for (int i = 0; i < 10; i++)
		fftw_execute(plan);
		std::cout << "Done (Fourier) Transforming" << std::endl;


		std::cout << "Writing"<< std::endl;


		// float max = 0;
		// float avg = 0;
		// float mag = 0;
		// for (int i = 0; i < SZ*SZ; i++){
		// 	mag = std::sqrt(C[i].x*C[i].x + C[i].y*C[i].y);
		// 	if (mag > max){
		// 		max = mag;
		// 	}
		// 	avg += mag;
		// }
		// avg /= SZ*SZ;

		// float stdr = 0;
		// for (int i = 0; i < SZ*SZ; i++){
		// 	mag = std::sqrt(C[i].x*C[i].x + C[i].y*C[i].y);
		// 	stdr += (avg - mag)*(avg - mag);
		// }

		// stdr = std::sqrt(stdr/(SZ*SZ-1));


		double cmax = 50000;//400*135000.0*135000.0;
		double cmin = 0;
		double abs = 0;

		// write to image, flip spectra
		for (int i = 0; i < SZ*SZ; i++){
			// std::cout << "i: " << i << std::endl;
			// abs = std::abs(C[i][0]);// + C[i][1]*C[i][1];
			abs = C[i][0]*C[i][0] + C[i][1]*C[i][1];
			if (i < 20 || (i < 2050 && i > 2040))// && i > 2048*1024-20)
				std::cout << "C[" <<i <<"] = " << C[i][0] << " + " << C[i][1] << "j"<< std::endl;
			// std::cout << "i: " << i << std::endl;

			unsigned int temp = static_cast<unsigned int> (999*((abs-cmin)/(cmax-cmin)));
			if (temp > 999){
				temp = 999;
			} else if (temp < 0){
				temp = 0;
			}
			// std::cout << "i: " << i << std::endl;
			rgb_t px_color = jet_colormap[temp];
			// std::cout << "i: " << i << std::endl;


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

		//Create and save image
		image.create(SZ, SZ, img); //create image from Uint8 array
		std::ostringstream oss;
		oss << std::setfill('0') << std::to_string(time) << std::setw(4);
		std::string title = "cppFrames/" + oss.str() + ".jpg";
		image.saveToFile(title);//"Frames6/" + std::to_string(kp) + img_extension);
	}


	// Garbage collection
	fftw_free(A);
	fftw_free(C);
	fftw_destroy_plan(plan);

	delete[] norm;
	delete[] data;
	fits_close_file(fptr, &status);
	return 0;
}



void cx_mult (fftw_complex &a, fftw_complex b){
	fftw_complex temp;
	temp[0] = a[0]*b[0]-a[1]*b[1];
	temp[1] = a[0]*b[1] + a[1]*b[0];
	a[0] = temp[0]; a[1] = temp[1];
}


