CC=nvcc
CXXFLAGS=-lcfitsio -std=c++11 -O2 -lpsrdada
NVFLAGS=-lcufft -Wno-deprecated-gpu-targets

all: real

debug: CXXFLAGS += -D DEBUG=1 -g -lineinfo -lsfml-graphics
debug: real

real: realtime_dada_dev.cu
	$(CC) -o $@ $^ $() $(CXXFLAGS) $(NVFLAGS)

clean: 
	rm real
