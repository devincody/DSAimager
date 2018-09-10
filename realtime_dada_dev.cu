// -*- c++ -*-
/* General Includes */
#include <iostream>
#include <complex>
#include <sstream>
#include <iomanip>
#include <string>
#include <stdio.h> // printf for device functions


/* Images */
#if DEBUG
	#include <SFML/Graphics.hpp>
	#include "bitmap_image.hpp" // colorscheme
#endif

/* Fits Files */
#include "fitsio.h" // fits files for fake data

/* Cuda Libraries */
#include <cufft.h>
#include <curand.h>
#include <cuda_runtime.h>

// nvcc -o real realtime_imager.cu -lsfml-graphics -lcfitsio -O2 -lcufft -std=c++11 -Wno-deprecated-gpu-targets

#include "realtime_dada.cuh"   


int main (int argc, char *argv[]){
	/* DADA defs */
	dada_hdu_t* hdu_in = 0;
	multilog_t* log = 0;
	int core = -1;
	int gpu = 1;
	key_t in_key = 0x0000dada;
	int observation_complete=0;
	uint64_t header_size = 0;

	/***************************************************
	Parse Command Line Options
	***************************************************/
	int arg = 0;
	while ((arg=getopt(argc,argv,"c:k:g:h")) != -1) {
		switch (arg) {
		// to bind to a cpu core
			case 'c':
				if (optarg){
					core = atoi(optarg);
					break;
				} else {
					printf ("ERROR: -c flag requires argument\n");
					return EXIT_FAILURE;
				}
				// to set the dada key
			case 'k':
				if (sscanf (optarg, "%x", &in_key) != 1) {
				fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
				return EXIT_FAILURE;
				}
				break;
			case 'g':
				if (optarg){
					gpu = atoi(optarg);
					break;
				} else {
					printf ("ERROR: -g flag requires argument\n");
					return EXIT_FAILURE;
				}
			case 'h':
				usage();
				return EXIT_SUCCESS;
		}
	}

	/***************************************************
	Initialize hdu
	***************************************************/

	// DADA stuff
	log = multilog_open ("real", 0);
	multilog_add (log, stderr);
	multilog (log, LOG_INFO, "creating hdu\n");

	// create dada hdu
	hdu_in	= dada_hdu_create (log);
	// set the input hdu key
	dada_hdu_set_key (hdu_in, in_key);

	// connect to dada buffer
	if (dada_hdu_connect (hdu_in) < 0) {
		printf ("could not connect to dada buffer\n");
		return EXIT_FAILURE;
	}

	// lock read on buffer
	if (dada_hdu_lock_read (hdu_in) < 0) {
		printf ("could not lock to dada buffer\n");
		return EXIT_FAILURE;
	}

	// Bind to cpu core
	if (core >= 0)
	{
		printf("binding to core %d\n", core);
		if (dada_bind_thread_to_core(core) < 0)
		printf("failed to bind to core %d\n", core);
	}

	multilog (log, LOG_INFO, "Done setting up buffer\n");

	/***************************************************
	Deal with Headers
	***************************************************/
	// read the headers from the input HDU and mark as cleared
	// will block until header is present in dada ring buffer
	char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
	if (!header_in)
	{
		multilog(log ,LOG_ERR, "main: could not read next header\n");
		dsaX_dbgpu_cleanup (hdu_in, log);
		return EXIT_FAILURE;
	}

	if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
	{
		multilog (log, LOG_ERR, "could not mark header block cleared\n");
		dsaX_dbgpu_cleanup (hdu_in, log);
		return EXIT_FAILURE;
	}

	// size of block in dada buffer
	uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
	uint64_t bytes_read = 0, block_id;
	char *block;

	multilog (log, LOG_INFO, "Done setting up header \n");
	
	std::cout << "block size is: " << block_size << std::endl;

	// start things
	// multilog(log, LOG_INFO, "starting observation\n");




	/***************************************************
	Physical Constants, offsets
	***************************************************/
	// int idx_usable_start = 300;
	// int idx_usable_end   = 1800;
	// int n_usable_idx = idx_usable_end - idx_usable_start;
	// float f_tot_end    = 1.53;
	float pos_extent =  1299.54002797;

	/***************************************************
	Datastructures for moving data
	***************************************************/

	/* Image Datastructures */
	#if DEBUG
		sf::Uint8 *img = new sf::Uint8[SZ*SZ*N_STREAMS*4]; // image data
		if (!img){ std::cerr << "img not allocated" << std::endl;}

		sf::Image image[N_STREAMS];						 // SFML image object
		std::ostringstream oss;
		std::string title;
		
	#endif
	cufftComplex *C = new cufftComplex[SZ*SZ*N_STREAMS]; //output image on device
	if (!C){std::cerr << "Matricies not allocated" << std::endl;}
	// cufftComplex *A = new cufftComplex[SZ*SZ];

	// for (int i =0; i < SZ*SZ; i++){
	// 	A[i].x = 1;
	// 	A[i].y = 0;
	// }

	/* gpu timers */
	float gpu_time_ms_solve = -1;
	//float avg_gpu_time = 0;
	//float avg_gpu_time_cnt = 0;

	/* Datastructures for Calcuations */
	cufftComplex *d_A, *d_C;
	float *d_data;

	int floats_per_freq = N_POL*N_COMPLEX; // number of floats per frequency (220)

	int floats_per_time = (N_CORRELATION_PRODUCTS)*N_CHANNELS_per_GPU/F_AVERAGING*N_POL*N_COMPLEX;
	int bytes_per_time = floats_per_time * sizeof(float);

	int floats_per_block = N_TIMESTEPS_per_BLOCK * floats_per_time;
	int bytes_per_block = N_TIMESTEPS_per_BLOCK * bytes_per_time;

	std::cout << "bytes per block: " << bytes_per_block << " should be: 42240000" << std::endl;

	gpuErrchk(cudaMalloc(&d_A,      SZ*SZ*N_STREAMS*sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc(&d_C,      SZ*SZ*N_STREAMS*sizeof(cufftComplex)));
	// gpuErrchk(cudaMalloc(&d_A_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	// gpuErrchk(cudaMalloc(&d_B_cal,  N_BASELINES*n_usable_idx*sizeof(cufftComplex) ));
	// gpuErrchk(cudaMalloc(&d_data,   n_floats_per_freq*n_usable_idx*N_TIMESTEPS*sizeof(float)));
	gpuErrchk(cudaMalloc(&d_data,   N_BLOCKS_on_GPU*bytes_per_block));

	/***************************************************
	Make Plans for FFT for each stream
	***************************************************/
	cufftHandle plan[N_STREAMS];
	cudaStream_t stream[N_STREAMS];
	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamCreate(&(stream[i])));
		gpuFFTchk(cufftPlan2d(&(plan[i]), SZ, SZ, CUFFT_C2C));
		gpuFFTchk(cufftSetStream(plan[i], stream[i]));
	}

	/***************************************************
	Pull .fits file data into memory
	***************************************************/
	// Variables
	// fitsfile *fptr;
	// int status = 0;
	// int ncols = 0;
	// long nrows = 0;
	
	// int column_number = 1;
	// int null_values;

	// // Collect information about the data
	// std::cout << "Opening data file" << std::endl;
	// fits_open_data(&fptr, "test.fits", READONLY, &status);
	// std::cout << "Opened data file" <<std::endl;
	// fits_get_num_rows(fptr, &nrows, &status);
	// fits_get_num_cols(fptr, &ncols, &status);
	// std::cout <<"rows: " << nrows <<" cols: " << ncols << std::endl;
	// std::cout << "Frequencies: " << n_usable_idx << std::endl;

	// // Allocate Host Memory for incoming data
	// float *data = new float[nrows];
	// gpuErrchk(cudaHostRegister(data, nrows*sizeof(float), cudaHostRegisterPortable));

	// // Read .fits file
	// fits_read_col(fptr, TFLOAT, column_number,
	// 			  1, 1, nrows, NULL, 
	// 			  data,
	// 			  &null_values , &status);

	// std::cout << "Done Reading" << std::endl;

	/***************************************************
	Calculate Constants 
	***************************************************/
	float c = 299792458.0;
	// float wavelength;
	int px_per_res_elem = 3;
	float mv = ((float) SZ)/(2.0*pos_extent)* c/(END_F*1E9)/px_per_res_elem;
	std::cout << "mv: " << mv <<std::endl;
	// std::cout << "pos_extent: " << pos_extent << std::endl;
	// std::cout << "other: " << c/(END_F*1E9)/px_per_res_elem << std::endl;

	/***************************************************
	Thrust constructs for finding maximum values
	***************************************************/

	/* Data Structures for Finding max values */
	cufftComplex thresh;
	thresh.x = 9000000.0;
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
	multilog (log, LOG_INFO, "Allocated Thrust Vectors\n");
	// Allocate host-side memory for values above thresh
	cufftComplex* h_value = new cufftComplex[N_DUMPPOINTS*N_STREAMS];
	int* h_index = new int[N_DUMPPOINTS*N_STREAMS];
	gpuErrchk(cudaHostRegister(h_value, N_DUMPPOINTS*N_STREAMS*sizeof(float), cudaHostRegisterPortable));
	gpuErrchk(cudaHostRegister(h_index, N_DUMPPOINTS*N_STREAMS*sizeof(float), cudaHostRegisterPortable));



	/***************************************************
	Start continuous Loop
	***************************************************/
	int time[N_STREAMS]; //describes which time step is being processed by each stream


	// std::cout << "hello: " << bytes_per_block<< std::endl;

	//Take ~1 second to fill up buffer on GPU
	for (int i = 0; i < N_BLOCKS_on_GPU; i++){
		multilog (log, LOG_INFO, "Filling GPU\n");
		block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
		std::cout << "Bytes Read: " << bytes_read << ", Should also be 42240000" << std::endl;
		if (bytes_read < block_size){
			std::cerr << "Could not fill GPU" << std::endl;
			return EXIT_FAILURE;
		} else {
			std::cout << "Read correct size" << std::endl;
		}

		gpuErrchk(cudaMemcpy(&(d_data[i*floats_per_block]), 
							 ((float *) block),
							 bytes_per_block,
							 cudaMemcpyHostToDevice));
		multilog(log, LOG_INFO, "Copied\n");
		ipcio_close_block_read (hdu_in->data_block, bytes_read);
	}


	bool done_with_block = false;
	bool written_image = false;
	int current_block = -1, status=0;
	uint64_t block_counter = N_BLOCKS_on_GPU;
	char foutnam[200];
	fitsfile *fptr;
	float outarr[SZ][SZ];
	
	// Assign each stream a time step to start
	for (int i = 0; i < N_STREAMS; i++){
		time[i] = i;
	}

	while(!observation_complete){
		#if DEBUG
			START_TIMER();
		#endif
		// Get pointer to dada buffer
		block = ipcio_open_block_read(hdu_in->data_block, &bytes_read, &block_id);

		if (bytes_read < block_size){
			multilog(log, LOG_INFO, "Not enough bytes\n");
			observation_complete = 1;
			// cudaProfilerStop();
			break;
		}

		#if DEBUG
			multilog(log, LOG_INFO, "Open new block for analysis\n");
		#endif



		if (current_block == N_BLOCKS_on_GPU -1){
			current_block = 0;
		} else {
			current_block++;
		}

		done_with_block = false;
		written_image = false;	      
		while (!done_with_block){ // Continuously iterate through streams until all new time steps are delt with.
			#if DEBUG
				std::cout << "time[0]: " << time[0] << ", stream: " << 0 << std::endl;
			#endif
			//Give each stream a task
			for (int s = 0; s < N_STREAMS; s ++){
				if (time[s] < N_TIMESTEPS_per_BLOCK){
					int img_strt = s*SZ*SZ;
					#if DEBUG
						std::cout << "time: " << time[0] << ", stream: " << s << std::endl;
					#endif
					// Asynchronously copy data to GPU
					gpuErrchk(cudaMemcpyAsync(&(d_data[current_block*floats_per_block + time[s] * floats_per_time]), 
											 ((float *) &(block[time[s] * floats_per_time])), 
											 bytes_per_time,
											 cudaMemcpyHostToDevice,
											 stream[s]));
					
					// Set gridded-visibility matrix to 0
					gpuErrchk(cudaMemsetAsync(&(d_A[img_strt]), 0, SZ*SZ*sizeof(cufftComplex), stream[s]));



				}
			}
			for (int s = 0; s < N_STREAMS; s++){
				if (time[s] < N_TIMESTEPS_per_BLOCK){
					int img_strt = s*SZ*SZ;
					int smem = 0;

					#if DEBUG
						std::cout << "calling kernel" << std::endl;
					#endif
					// gpuErrchk(cudaStreamSynchronize(stream[s]));

					// gpuErrchk
					grid<<<N_BASELINES, N_CHANNELS_per_GPU/F_AVERAGING, smem, stream[s]>>>(&(d_A[img_strt]), d_data, (N_TIMESTEPS_per_BLOCK*current_block + time[s]), gpu, 56.7712);
					// gpuErrchk(cudaMemcpyAsync(&(d_A[img_strt]), 
					// 						 A, 
					// 						 SZ*SZ*sizeof(cufftComplex),
					// 						 cudaMemcpyHostToDevice,
					// 						 stream[s]));

					#if DEBUG
						std::cout << "Done Filling Matrix" << std::endl;
					#endif


					/* Execute FFT */
					gpuFFTchk(cufftExecC2C(plan[s], &(d_A[img_strt]), &(d_C[img_strt]), CUFFT_FORWARD));
					

					#if DEBUG
					    std::cout <<"img start = " << img_strt << std::endl;

					    gpuErrchk(cudaMemcpyAsync(&(C[img_strt]), &(d_C[img_strt]), sizeof(cufftComplex)*SZ*SZ, cudaMemcpyDeviceToHost, stream[s]));
						multilog(log, LOG_INFO, "Executed FFT\n");
					#endif


					
					/* Find Largest Values */
					int n_vals = thrust::copy_if(thrust::cuda::par.on(stream[s]),
								make_zip_iterator(make_tuple(t_data[s], make_counting_iterator(0u))),
								make_zip_iterator(make_tuple(t_data[s], make_counting_iterator(0u))) + SZ*SZ,
								t_data[s],
								make_zip_iterator(make_tuple(d_value[s].begin(), d_index[s].begin())),
								greater_than_val<cufftComplex>(thresh)
								) - make_zip_iterator(make_tuple(d_value[s].begin(), d_index[s].begin()));

					#if DEBUG
						std::cout << "number of points = " << n_vals << std::endl;
						multilog(log, LOG_INFO, "Found largest Vals\n");
					#endif

					/* Copy Largest Values to RAM */
					if (n_vals < N_DUMPPOINTS && n_vals > 0){
						gpuErrchk(cudaMemcpyAsync(&(h_value[N_DUMPPOINTS*s]), thrust::raw_pointer_cast(d_value[s].data()), n_vals*sizeof(cufftComplex), cudaMemcpyDeviceToHost, stream[s]));
						gpuErrchk(cudaMemcpyAsync(&(h_index[N_DUMPPOINTS*s]), thrust::raw_pointer_cast(d_index[s].data()), n_vals*sizeof(int), cudaMemcpyDeviceToHost, stream[s]));
					}



					

					/* Calculate statistics of largest values */

					float maxx = 0;
					int indexx = 0;
					for (int i = 0; i < min(n_vals, N_DUMPPOINTS); i++){
						if (h_value[i + N_DUMPPOINTS*s].x > maxx){
							maxx = h_value[i + N_DUMPPOINTS*s].x;
							indexx = h_index[i + N_DUMPPOINTS*s];
						}
					}

					#if DEBUG
						std::cout << "time: " << time[s] << ", max value by copy_if = " << maxx << ", at location = " << indexx << std::endl;
					#endif
					/*
					TODO:
					Calculate coincidences (std::map)
					At end, search through list of all arrays, if proximate data points, include in sum.
					*/


					// write image to disk if we want
					// if conditions to be replaced with reasonable statistic
					if (!written_image) {
					  if ((s==0) && (((block_counter % 5) == 0) || (block_counter == 4))) {
					  // if (n_vals > 0){

					    // copy to host
					    gpuErrchk(cudaMemcpyAsync(&(C[img_strt]), &(d_C[img_strt]), sizeof(cufftComplex)*SZ*SZ, cudaMemcpyDeviceToHost, stream[s]));
					    
					    // write fits image					    
					    sprintf(foutnam,"/mnt/nfs/test_image_dir/image_%lu.fits",block_counter);

					    #if DEBUG
						    std::cout << "Writing fits file: " << foutnam << std::endl;
						#endif

					    long naxes[2] = {SZ, SZ};
					    long naxis = 2;
					    
					    fits_create_file(&fptr, foutnam, &status);
					    if (status) std::cerr << "create_file FITS error " << status << std::endl;
					    fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status);
					    if (status) std::cerr << "create_img FITS error " << status << std::endl;					    
					    
					    // write file
					    for (int i=0;i<SZ*SZ;i++) {
						    int idx;
						      
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
					      // idx = i;
					      outarr[(int)(i/SZ)][(int)(i % SZ)] = C[idx + img_strt].x;
					      
					    }
					    
					    fits_write_img(fptr, TFLOAT, 1, SZ*SZ, outarr[0], &status);
					    fits_close_file(fptr, &status);
					    if (status) std::cerr << "final FITS error " << status << std::endl;	
					    
					    written_image = true;
					  }
					} // end write image

					
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

						float cmax = 200000.0;
						float cmin = 0;
						float abs = 0;


						int idx;


						// write to image, flip spectra
						for (int i = 0; i < SZ*SZ; i++){
							abs = C[i+img_strt].y; //std::sqrt(C[i+img_strt].x*C[i+img_strt].x);// + C[i+img_strt].y*C[i+img_strt].y);
							unsigned int temp = static_cast<unsigned int> (999*((abs-cmin)/(cmax-cmin)));
							if (temp > 999){temp = 999;}

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
							// idx = i;
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
					


					time[s] += N_STREAMS;
				} // END IF (time[s] < N_TIMESTEPS_per_BLOCK)
			}// END FOR (int s = 0; s < N_STREAMS; s ++)

			/* Calculate whether a new block is needed */
			#if DEBUG
				multilog(log, LOG_INFO, "Checking Values\n");
			#endif		
			int count = 0;
			for (int s = 0; s < N_STREAMS; s ++){
				if (time[s] >= N_TIMESTEPS_per_BLOCK){
					#if DEBUG
						std::cout << "time[" << s << "] = " << time[s] << std::endl;
					#endif
					count ++;
				}
			}
			if (count == N_STREAMS){
				done_with_block = true;
			}

			/* If current block is done, need to give new assignments */
			if (done_with_block){
				std::cout << "DONE WITH BLOCK" << std::endl;
				ipcio_close_block_read (hdu_in->data_block, bytes_read);
				block_counter++;
				for (int j = 0; j < N_STREAMS; j++){
					time[j] -= N_TIMESTEPS_per_BLOCK;
				}
			}

		}

		
		#if DEBUG
			STOP_RECORD_TIMER(gpu_time_ms_solve);// avg_gpu_time += gpu_time_ms_solve; avg_gpu_time_cnt += 1;
			std::cout << "Done (Fourier) Transforming in " << gpu_time_ms_solve/N_TIMESTEPS_per_BLOCK <<" ms \n" << std::endl;
		#endif
	}
	// Garbage collection
	#if DEBUG
		delete[] C;
	#endif

	gpuErrchk(cudaFree(d_A));
	gpuErrchk(cudaFree(d_C));
	gpuErrchk(cudaFree(d_data));
	// gpuErrchk(cudaFree(d_A_cal));
	// gpuErrchk(cudaFree(d_B_cal));
	// gpuFFTchk(cufftDestroy(plan));

	for (int i = 0; i < N_STREAMS; i++){
		gpuErrchk(cudaStreamDestroy(stream[i]));
		gpuFFTchk(cufftDestroy(plan[i]));
	}

	// delete[] norm;
	// gpuErrchk(cudaHostUnregister(data));
	gpuErrchk(cudaHostUnregister(h_value));
	gpuErrchk(cudaHostUnregister(h_index));
	// delete[] data;
	delete[] h_value;
	delete[] h_index;
	dsaX_dbgpu_cleanup(hdu_in, log);
	// fits_close_file(fptr, &status);
	return 0;
}
































