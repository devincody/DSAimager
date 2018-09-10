## Why does this folder exist?
I initially decided not to use git to track files. Now that I'm using git, this folder contains a backups of my code at various stages.

## (Rough) Chronology and (Approximate) Descriptions
0. ../Python/fftVolt.py 
    * Proof of concept code. 
    * Imaged with raw-voltages (beam-formed) or visibilities (imaging)
    *  Could use made-up data or actual data.
1. ../Python/fftVolt2.py
    * Worked fully.
2. ../Python/fftVolt_withCal.py
    * Included baseline-based calibrators for the visibilities.
3. vis.cpp
    * Initial port of python implementation to C++.
    * Used Armadillo for FFT. 
    * Reading data from .fits files.
4. vis2.cpp
    * Replaced Armadillo with FFTW.
5. vis2_withCal.cpp
    * Included baseline-based calibrators for the visibilities.
6. gpu_vis.cu
    * Replaced FFTW with CuFFT. 
    * Still gridding on CPU.
7. gpu_vis.cu
    * Gridding moved to GPU.
8. gpu_vis_streamed.cu
    * Enabled streams for asynchronous execution on GPU for higher throughput. 
    * Uses thrust::copy_if() to analyze images.
9. realtime_imager.cu
    * Made several performance upgrades.
10. realtime_imager2.cu
    * Included #if debug statements to remove any extraneous calculations or print statements. 
    * Speed up gridding step by enabling a single thread to handle each frequency.
101 realtime_dada.cu
    * Data is no longer read from .fits files, but instead is read from dada buffers. 
    * Indexing of input data is changed
12. realtime_dada_dev.cu
    * More performance optimizations.
    * included ability to write images to .fits files.
    * fixed the gridding step which was causing high frequencies to be mapped to low frequencies and low frequencies to be mapped to high frequencies.
  
