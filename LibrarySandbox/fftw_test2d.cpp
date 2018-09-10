#include <iostream>
#include <fftw3.h>
#include <armadillo>

#define N 2048*4

int main(){
	int Nsq = N*N;
		
	fftw_complex *data, *odata, *oodata;

	data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N*N);
	odata = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*N*N);
	oodata = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*N*N);

	fftw_plan plan, blan;

	plan = fftw_plan_dft_2d(N, N, data, odata, FFTW_FORWARD, FFTW_ESTIMATE);
	blan = fftw_plan_dft_2d(N, N, odata, oodata, FFTW_BACKWARD, FFTW_ESTIMATE);

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			data[i*N+j][0] = i+j + .01*j;
			data[i*N+j][1] = i-j + 1.112*i;
		}
	}

	fftw_execute(plan);
	fftw_execute(blan);

	int i = 7531;
	for (int j = 0; j < 20; j++){
		std::cout << "oodata[" << i << "][" << j <<"] = \t" << oodata[i*N+j][0]/Nsq <<" + " << oodata[i*N+j][1]/Nsq << "j vs. \t" << data[i*N+j][0] << " + " << data[i*N+j][1] << "j"  << std::endl;
	}

	fftw_destroy_plan(plan);
	fftw_destroy_plan(blan);

	fftw_free(data); fftw_free(odata); fftw_free(oodata);

	return 0;

}

