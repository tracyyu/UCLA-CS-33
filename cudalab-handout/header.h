#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#define CUDA
#endif

#define RAND01() ((float)random()/(float)RAND_MAX)

// PARAMETERS
#define GYRO_DT       0.01f
#define ACCL_DT       0.01f
#define STDDEV_ODOVel 0.1f
#define STDDEV_GYRO   0.1f
#define STDDEV_POS    0.1f
#define STDDEV_VEL    0.02f

////////////////////////////////////////////////////////////////////////////////

void entry_type1( \
         int N, float seed, float rand1[], float mcl_out[], int ws_out[], \
         float quat[][4], float pos[][3], float vel[][3], \
         float quat2[][4], float pos2[][3], float vel2[][3]);

void entry_type2( \
         int N, float x, float y, float z, float quat[][4], float randn3[][3]);

void entry_type3( \
         int N, float a0, float a1, float a2, float seed, \
         float randn3[][3], float rand1[], float mcl_out[], int ws_out[], \
         float quat[][4], float pos[][3], float vel[][3], \
         float quat2[][4], float pos2[][3], float vel2[][3]);

void entry_type4( \
         int N, float d0, float d1, float d2, float d6, float d7, float seed, \
         float randn3[][3], float rand1[], float mcl_out[], int ws_out[], \
         float quat[][4], float pos[][3], float vel[][3], \
         float quat2[][4], float pos2[][3], float vel2[][3]);

#ifdef CUDA
#define CUDA_CHECK(msg) \
	do { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) { \
			fprintf(stderr, "CUDA Error: %s: %s.\n", \
			        (msg), cudaGetErrorString(err)); \
			exit(-1); \
		} \
	} while(0)
#endif

void allocateData(int,float**,float**,float**, \
                      float**,float**,float**, \
                      float**,float**,float**, \
		      int**,float**);

void initializeData(int,float,float[][3],float[][3], \
                    float[][4],float[][3],float[][3]);

void deallocateData(float*,float*,float*,float*,float*,float*,
                    float*,float*,float*,int*,float*);

////////////////////////////////////////////////////////////////////////////////

// TIMING
typedef struct {
	struct timeval t;
	double seconds;
} Time;

#define TIMING_INIT(name) \
	Time name = {{0},0}

#define TIMING_RESET(Time) \
	Time.seconds = 0

#define TIMING_PRINT(Time) \
	printf("%s: %f sec\n", #Time, Time.seconds)

#define TIMING_START(Time) \
	gettimeofday(&Time.t,NULL)

#define TIMING_STOP(Time) \
	do { struct timeval t2; \
	     gettimeofday(&t2,NULL); \
	     Time.seconds += (double)(t2.tv_sec  - Time.t.tv_sec ) + \
	                1e-6*(double)(t2.tv_usec - Time.t.tv_usec);  \
	} while(0)

////////////////////////////////////////////////////////////////////////////////

#endif // LOCALIZATION_H
