# DSAimager

## What is the Deep Synoptic Array (DSA)?
DSA is a 10-element radio interferometer located at the Owens Valley Radio Observatory (OVRO) in California. The purpose of this array is to detect and localize enignmatic pulses of radio energy known as fast radio bursts (FRBs). If you're interested in learning more about radio interferometers, check out my blog post about how they work [here](https://devincody.github.io/Blog/2018/02/27/An-Introduction-to-Radio-Interferometry-for-Engineers.html). 

## What does this code do?
This project is a collection of gpu-accelerated code to do the actual imaging and localization of FRBs. As an example of the capabilities of this project, the following .gif shows an imaged pulse coming from the pulsar at the heart of the crab nebula.

![Crab Pulsar](https://github.com/devincody/DSAimager/blob/master/Images/pulse.gif)

The code operates in realtime which means that it must produce an image of the radio sky every 1.04 milliseconds. 

## How does it work?
Data from the antennas is collected in dada ring buffers before being transfered to the GPU. A GPU kernel is executed to fill a visibility matrix and then cufft is used to fourier transform the data. The statistics of the resulting image are then analyzed with various thrust functions. If those statistics meet some threshold, then the data is dumped to disk for further analysis.

## How do I run the code?
make

dada_db -k <name, e.g., dada> -n <number of blocks, e.g., 8> -b <number of bytes per block, which should be 42240000 = 125 chans * 2 pols * 2 complex * 384 time samples * sizeof(float)> -l -p 

./real -k <name, e.g., dada> -c <cpu number> -g <gpu number>
 
 dada_junkdb -c 0 -k <name, e.g., dada> -r <dada rate in MB/s, which should be 105> -t <seconds to run for, e.g., 500> /mnt/nfs/code/dsaX/src/correlator_header_dsaX.txt

e.g.
dada_db -k baab -n 8 -b 42240000 -l -p 

./real -k baab -c 0 -g 0

 dada_junkdb -c 0 -k baab -r 105 -t 5 /mnt/nfs/code/dsaX/src/correlator_header_dsaX.txt
  
## How do I monitor the code?
### dada buffers
dada_dbmonitor -k <name, e.g., dada>
### CUDA execution
nvvp

## How do I kill the code?
ctrl-C everything
dada_db -k <name, e.g., dada> -d
