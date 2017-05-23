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

/** Data Structures **/
typedef struct {
    int width;
    int height;
    int data[];
} I2D;

typedef struct {
    int width;
    int height;
    float data[];
} F2D;

#define subsref(a,i,j) (a)->data[(i) * (a)->width + (j)]
#define asubsref(a,i)  (a)->data[i]
#define arrayref(a,i)  (a)[i]

/** Memory Management **/
I2D* iMallocHandle(int rows, int cols);
F2D* fMallocHandle(int rows, int cols);
void iFreeHandle(I2D* out);
void fFreeHandle(F2D* out);
I2D* iSetArray(int rows, int cols, int val);
F2D* fSetArray(int rows, int cols, float val);
F2D* fDeepCopy(F2D* in);
F2D* fDeepCopyRange(F2D* in, int startRow, int numberRows, int startCol, int numberCols);

/** Math Operations **/
F2D* fDivide(F2D* a, float b);
F2D* fMdivide(F2D* a, F2D* b);
F2D* ffDivide(F2D* a, F2D* b);
F2D* fTimes(F2D* a, F2D* b);
F2D* fMtimes(F2D* a, F2D* b);
F2D* fMinus(F2D* a, F2D* b);
F2D* fPlus(F2D* a, F2D* b);
F2D* fHorzcat(F2D* a, F2D* b);
F2D* fSum2(F2D* inMat, int dir);
F2D* fSum(F2D* inMat);

/** Miscellaneous **/
F2D* randnWrapper(int m, int n);
F2D* randWrapper(int m, int n);

/** Localization **/
I2D* weightedSample(F2D* w);
void generateSample(F2D *w, F2D *quat, F2D *vel, F2D *pos);
F2D* get3DGaussianProb( F2D* data, F2D* mean, F2D* A);
F2D* mcl(F2D* x, F2D* sData, F2D* invConv);
F2D* eul2quat(F2D* angle);
F2D* quat2eul(F2D* quat);
F2D* quatConj(F2D* a);
F2D* quatMul(F2D* a, F2D* b);
F2D* quatRot(F2D* vec, F2D* rQuat);

void entry_type1(F2D *sData, F2D *quat, F2D *pos, F2D *vel);
void entry_type2(F2D *sData, F2D *ones, F2D **quat, F2D *randW);
void entry_type3(F2D *sData, F2D *ones, F2D *quat, F2D **pos, F2D **vel);
void entry_type4(F2D *sData, F2D *ones, F2D *quat, F2D *pos, F2D *vel,
                 F2D *STDDEV_GPSPos, F2D *randW);

/** Timing **/
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

/** Parameters **/
#define gyroTimeInterval 0.01
#define acclTimeInterval 0.01
#define STDDEV_GPSVel 0.5
#define STDDEV_ODOVel 0.1
#define STDDEV_ACCL 1
#define M_STDDEV_GYRO 0.1
#define M_STDDEV_POS 0.1
#define M_STDDEV_VEL 0.02

#endif /* LOCALIZATION_H */
