.PHONY: cuda omp run clean

SRC ?= localization
PROG ?= ./localization
IN_DATA ?= input.txt
VALID_DATA ?= correct.txt

NVCC_FLAGS = --compiler-options -fno-strict-aliasing --compiler-options -fno-inline

FLAGS=-o $(PROG) -I. -O3
LIBS=-lm

N ?= 8000
SEED ?= 12345

ifdef EMU
FLAGS+=-deviceemu
endif

ifdef NOVERIFY
FLAGS+=-DNOVERIFY
endif

ifdef GENERATE
FLAGS+=-DGENERATE
endif

cuda:
	ln -fs main.c main.cu
	nvcc $(FLAGS) $(NVCC_FLAGS) $(SRC).cu main.cu $(LIBS) 
	rm main.cu
omp:
	gcc -Wall -fopenmp $(FLAGS) $(SRC).c main.c $(LIBS)
run:
	$(PROG) $(N) $(SEED) $(IN_DATA) $(VALID_DATA)
clean:
	rm -f main.cu core.* $(PROG)
