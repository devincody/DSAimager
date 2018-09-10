#include <iostream>
#include <armadillo>
#include "fitsio.h"
// #include <cstdlib>
#include <SFML/Graphics.hpp>

// g++ test.cpp -o test -larmadillo -lcfitsio -lsfml-graphics

using namespace arma;

#define IDX(t,f,n) ((t*220*2048) + (f*220) + n)
// Indexed by time, frequency, baseline (in order of largest to smallest jumps)

int main(){
	sf::Image img;
	fitsfile *fptr;
	char card[FLEN_CARD];
	int status = 0, nkeys, ii;

	fits_open_file(&fptr, "../test.fits", READONLY, &status);
	fits_get_hdrspace(fptr, &nkeys, NULL, &status);

	for (ii = 1; ii < nkeys; ii++){
		fits_read_record(fptr, ii, card, &status);
		std::cout << "record: " << card << std::endl;
	}

	float tsamp;
	fits_close_file(fptr, &status);

	fits_open_data(&fptr, "../test.fits", READONLY, &status);
	fits_read_key(fptr, TFLOAT, "TSAMP", &tsamp, NULL, &status);
	std::cout << "hello! " << tsamp << std::endl;
	fits_read_key(fptr, TFLOAT, "MJD", &tsamp, NULL, &status);
	std::cout << "hello! " << tsamp << std::endl;

	char* antennas;
	antennas = (char *) malloc(20*sizeof(char));
	if (antennas == NULL) exit(1);

	std::cout << "malloced" << std::endl;

	fits_read_key(fptr, TSTRING, "ANTENNAS", antennas, NULL, &status);
	std::cout << "read" << std::endl;
	std::cout << "antenna order: " << antennas << std::endl;

	int ncols = 0;
	long nrows = 0;
	fits_get_num_rows(fptr, &nrows, &status);
	fits_get_num_cols(fptr, &ncols, &status);

	std::cout <<"rows: " << nrows <<" cols: " << ncols << std::endl;

	float *data = (float *) malloc(nrows*sizeof(float));
	data[1000] = 123.15;
	int column_number = 1;
	int null_values;

	fits_read_col(fptr, TFLOAT, column_number,
				  1, 1, nrows, NULL, 
				  data,
				  &null_values , &status);

	// for (int i = 0; i < 1000; i++){
	// 	std::cout << data[i]<< ", ";
	// }
	// std::cout << std::endl;

	float* slice = &(data[IDX(231,5,0)]);
	for (int i = 0; i < 220; i++){
		std::cout << slice[i]<< ", ";
	}
	std::cout << std::endl;

	for (int i = 0; i < 180; i++){
		std::cout << data[IDX(231,203,i)]<< ", ";
	}
	std::cout << " == " << IDX(231,0,1) << " == " << std::endl;



	// data = fptr.dat[1,:];

	// for (int i = 1000; i < 1500; i++){
	// 	std::cout << data[i];
	// }
	// std::cout << std::endl;


	// int colnum = 0;
	// char r = '*';
	// // fits_get_colname(fptr, CASEINSEN, &r, &colnum, &status);
	// std::cout << "col num: " << colnum << std::endl;

	free(antennas);
	fits_close_file(fptr, &status);

	// std::cout << "end" << std::endl;

	// cx_vec X(10);
	// std::cout << "end" << std::endl;

	// for (int i = 0; i < 10; ++i){
	// 	if (i < 5){
	// 		X(i) = i;
	// 	} else {
	// 		X(i) = 9-i;
	// 	}
	// }

	// X.print("X:");

	// cx_vec Z, Y;
	// Z = fft(X);
	// Z.print("Z:");
	// Y = ifft(Z);

	// Y.print("Y:");


	// std::cout << "end" << std::endl;

	// cx_mat X(8,8);
	// std::cout << "end" << std::endl;

	// for (int i = 0; i < 8; ++i){
	// 	for (int j = 0; j < 8; ++j){
	// 		if (i+j < 7){	
	// 			X(i,j) = i+j;
	// 		} else {
	// 			X(i,j) = 14-i-j;
	// 		}
	// 	}
	// }

	// real(X).print("X:");

	// cx_mat Z, Y;
	// Z = fft2(X);
	// Z.print("Z:");
	// Y = ifft2(Z);

	// real(Y).print("Y:");



}














