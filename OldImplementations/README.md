## Why does this folder exist?
I initially decided not to use git to track files. Now that I'm using git, this folder contains a backups of my code at various stages.

## (Rough) Chronology and (Approximate) Descriptions
-2. ../Python/fftVolt.py 
    1. Proof of concept code. 
    2. Imaged with raw-voltages (beam-formed) or visibilities (imaging), with made-up data or actual data.
-1. ../Python/fftVolt2.py
    * Worked fully.
0. ../Python/fftVolt_withCal.py
    * Included baseline-based calibrators for the visibilities.
1. vis.cpp
    1. Initial port of python implementation to C++.
    2. Used Armadillo for FFT. 
    3. Reading data from .fits files.
2. vis2.cpp
  1. Replaced Armadillo with FFTW.
3. vis2_withCal.cpp
  1. Included baseline-based calibrators for the visibilities.
4. gpu_vis.cu
  1. Replaced FFTW with CuFFT. Still gridding on CPU.
5. gpu_vis.cu
  1. Gridding moved to GPU.
6. gpu_vis_streamed.cu
  1. Enabled streams for asynchronous execution on GPU for higher throughput. Uses thrust::copy_if() to analyze images.
7. realtime_imager.cu
  * Made several performance upgrades
8. realtime_imager2.cu
  * Included #if debug statements to remove any extraneous calculations or print statements. Speed up gridding step by enabling a single thread to handle each frequency.
9. realtime_dada.cu
  * Data is no longer read from .fits files, but instead is read from dada buffers. Indexing of input data is changed
10. realtime_dada_dev.cu
  * Performance enhancements, included ability to write images to .fits files, fixed the gridding step which was causing high frequencies to be mapped to low frequencies and low frequencies to be mapped to high frequencies
  
