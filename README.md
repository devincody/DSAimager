# DSAimager

## What is the Deep Synoptic Array (DSA)?
DSA is a 10-element radio interferometer located at the Owens Valley Radio Observatory (OVRO) in California. The purpose of this array is to detect and localize enignmatic pulses of radio energy known as fast radio bursts (FRBs). If you're interested in learning more about radio interferometers, check out my blog post about how they work [here](https://devincody.github.io/Blog/2018/02/27/An-Introduction-to-Radio-Interferometry-for-Engineers.html). 

## What does this code do?
This project is a collection of gpu-accelerated code to do the actual imaging and localization of FRBs. As an example of the capabilities of this project, the following .gif shows an imaged pulse coming from the pulsar at the heart of the crab nebula.

![Crab Pulsar](https://github.com/devincody/DSAimager/blob/master/Images/pulse.gif)

The code operates in realtime which means that it must produce an image of the radio sky every 1.04 milliseconds. 

## How does it work?
Data from the antennas is collected in dada ring buffers before being transfered to the GPU. A GPU kernel is executed to fill a visibility matrix and then cufft is used to fourier transform the data. The statistics of the resulting image are then analyzed with various thrust functions. If those statistics meet some threshold, then the data is dumped to disk for further analysis.
