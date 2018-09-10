#include <iostream>
#include <complex>
#include <fstream>
#include <string>
#include <fftw3.h>

void cx_mult (fftw_complex a, fftw_complex b, fftw_complex &c);


int main(){
	std::cout <<  "hello" << std::endl;
	
	std::ifstream A_real, A_imag, B_real, B_imag;
	A_real.open("AA_real.txt");
	A_imag.open("AA_imag.txt");
	B_real.open("BB_real.txt");
	B_imag.open("BB_imag.txt");

	double tempa, tempb;
	fftw_complex* A_cal = new fftw_complex[45*1500];
	fftw_complex* B_cal = new fftw_complex[45*1500];
	fftw_complex a = {1,4};
	fftw_complex b = {3,-5};
	fftw_complex c;

	for (int i = 0; i < 45; i++){
		for (int j = 0; j < 1500; j++){
			A_real >> A_cal[i*1500 + j][0];
			A_imag >> A_cal[i*1500 + j][1];
			B_real >> B_cal[i*1500 + j][0];
			B_imag >> B_cal[i*1500 + j][1];
		}
	}

	// for (int i = 0; i < 45; i++){
	// 	for (int j = 0; j < 1500; j++){
	// 		std::cout << "A_cal[" << i*1500+j << "] = " << A_cal[i*1500 + j][0] << " + " << A_cal[i*1500 + j][0] << "j" << std::endl;
	// 	}
	// }
	cx_mult(a,b,c);
	cx_mult(fftw_complex(1,2),fftw_complex(3.1,-22),c);
	std::cout << "ans = " << c[0] << " + " << c[1] << "j" << std::endl;

	delete[] A_cal;
	delete[] B_cal;
	return 0;

}

void cx_mult (fftw_complex a, fftw_complex b, fftw_complex &c){
	c[0] = a[0]*b[0]-a[1]*b[1];
	c[1] = a[0]*b[1] + a[1]*b[0];
}




