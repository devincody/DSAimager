#include <iostream>
#include <fftw3.h>
#include <armadillo>

#define N 256

int main(){
	std::cout << "hello " << std::endl;
	fftw_complex *in, *out, *xout;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
	out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*N);
	xout = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*N);

	fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_back = fftw_plan_dft_1d(N, out, xout, FFTW_BACKWARD, FFTW_ESTIMATE);


	for (int i = 0; i < N/2; i++){
		in[i][0] = i;
		in[i][1] = 2.31 + ((float) i)/40;
	}


	for (int i = N/2; i < N; i++){
		in[i][0] = N - i;
		in[i][1] = -5.32 - ((float) i)/35;
	}

	fftw_execute(plan);
	fftw_execute(plan_back);

	fftw_destroy_plan(plan);
	fftw_destroy_plan(plan_back);

	for (int i = 0; i < N; i++){
		std::cout << "xout[" <<i <<"] = " <<xout[i][0]/N << " + " <<xout[i][1]/N << "j  vs. "<<in[i][0] << " + " <<in[i][1] << "j " << std::endl;
	}

	// for (int i = N/2; i < N; i++){
	// 	std::cout << "xout[" <<i <<"] = " <<xout[i][0]/N << " + " <<xout[i][1]/N << "j  vs. "<< N-i << " - 5.32j" << std::endl;
	// }

	fftw_free(in); fftw_free(out);
	fftw_free(xout);

	return 0;

}



