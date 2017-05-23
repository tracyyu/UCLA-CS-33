#include "header.h"

////////////////////////////////////////////////////////////////////////////////

float*
readInputFile(const char* fileName, int* rows_out, int* cols_out)
{
	int n, i, j, rows=0, cols;
	FILE* fp = fopen(fileName, "r");
	if(!fp) return NULL;
	float* fill = NULL;

	n = fscanf(fp, "%d", cols_out);
	cols = *cols_out;
	if (n != 1) goto done;

	n = fscanf(fp, "%d", rows_out);
	rows = *rows_out;
	if (n != 1) goto done;

	fill = (float*) malloc(rows*cols*sizeof(float));

	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		n = fscanf(fp, "%f", &fill[i*cols+j]);
		if (n != 1) { free(fill); fill = NULL; goto done; }
	}
done:
	fclose(fp);    
	return fill;
}

////////////////////////////////////////////////////////////////////////////////

void
readSensorData(float data[8], int* type, float inputData[], int cols, int* row)
{
	int i, k = *row;
	int t = (int) inputData[k*cols+1];

	if (t == 1 || t == 2 || t == 3)
	{
		for (i = 0; i < 3; i++)
			data[i] = inputData[k*cols + i+2];
		for (i = 3; i < 8; i++)
			data[i] = 0.f;
	}
	else if (t == 4)
	{
		for (i = 0; i < 3; i++)
			data[i] = inputData[k*cols + i+2];
		for (i = 3; i < 8; i++)
			data[i] = inputData[(k+1)*cols + i-3];
		k++;
	}
	else
	{
		fprintf(stderr, "Invalid input entry type %d\n", t);
		exit(-1);
	}

	*row = k + 1;
	*type = t;
}

////////////////////////////////////////////////////////////////////////////////

inline void
swap_ptrs(float** p1, float** p2)
{
	float* tmp = *p1;
	*p1 = *p2;
	*p2 = tmp;
}

////////////////////////////////////////////////////////////////////////////////

void verifyResult(FILE* validFile, int N, int row, int type, \
                  float* quat_d, float* pos_d, float* vel_d)
{
#ifdef CUDA
	int N4B = N*4*sizeof(float);
	int N3B = N*3*sizeof(float);
	float* quat = (float*) malloc(N4B);
	float* pos  = (float*) malloc(N3B);
	float* vel  = (float*) malloc(N3B);
	cudaMemcpy(quat, quat_d, N4B, cudaMemcpyDeviceToHost);
	cudaMemcpy( pos,  pos_d, N3B, cudaMemcpyDeviceToHost);
	cudaMemcpy( vel,  vel_d, N3B, cudaMemcpyDeviceToHost);
	CUDA_CHECK("verifyResult cudaMemcpy");
#else
	#define quat quat_d
	#define pos  pos_d
	#define vel  vel_d
#endif
	int i;
	float quatSum = 0.f, velSum = 0.f, posSum = 0.f;

	for (i = 0; i < 4*N; i++)
		quatSum += quat[i];
	for (i = 0; i < 3*N; i++)
		velSum += vel[i];
	for (i = 0; i < 3*N; i++)
		posSum += pos[i];
#ifdef CUDA
	free(quat);
	free(pos);
	free(vel);
#endif
	union { float f; unsigned u; } valid[3];

#ifdef GENERATE
	valid[0].f = quatSum;
	valid[1].f = velSum;
	valid[2].f = posSum;
	fprintf(validFile, "0x%08X 0x%08X 0x%08X\n", valid[0].u, valid[1].u, valid[2].u);
#else
	fscanf(validFile,"%X %X %X\n", \
	       &valid[0].u, &valid[1].u, &valid[2].u);

	if (fabsf((quatSum - valid[0].f)/valid[0].f) <= 0.01f &&
	    fabsf(( velSum - valid[1].f)/valid[1].f) <= 0.01f &&
	    fabsf(( posSum - valid[2].f)/valid[2].f) <= 0.01f)
	{
		// Correct!
	}
	else
	{
		fprintf(stderr, "Invalid result at row %d (type %d)!\n", row, type);
		fprintf(stderr, "   Your values: %f %f %f\n", \
			quatSum, velSum, posSum);
		fprintf(stderr, "Correct values: %f %f %f\n", \
			valid[0].f, valid[1].f, valid[2].f);
		exit(-1);
	}
#endif
}
////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char* argv[])
{
	// Check arguments
	if(argc < 5) {
		fprintf(stderr, "Usage: %s N SEED INFILE VALIDFILE\n", argv[0]);
		return -1;
	}

	int N = atoi(argv[1]);
	if (N < 1) {
		fprintf(stderr, "N must be greater than 0.\n");
		return -1;
	}

	srandom(atol(argv[2]));

	// Open input files
	int in_rows, in_cols;
	float* inputData = readInputFile(argv[3], &in_rows, &in_cols);
	if (!inputData) {
		fprintf(stderr, "Error opening `%s`.\n", argv[3]);
		return -1;
	}

#ifdef GENERATE
	FILE* validFile = fopen(argv[4],"w");
#else
	FILE* validFile = fopen(argv[4],"r");
#endif
	if (!validFile) {
		fprintf(stderr, "Error opening `%s`.\n", argv[4]);
		return -1;
	}

	// Configure OpenMP
#ifdef _OPENMP
	omp_set_num_threads(4);
#endif

	// Create data structures
	int row;
	float *randn3=NULL, *rand3=NULL, *rand1=NULL;
	float *quat=NULL, *vel=NULL, *pos=NULL;
	float *quat_new=NULL, *vel_new=NULL, *pos_new=NULL;
	int *ws_out=NULL;
	float *mcl_out=NULL;

#define F4 float(*)[4]
#define F3 float(*)[3]

	allocateData(N, &randn3, &rand3, &rand1, &quat, &vel, &pos, \
	                &quat_new, &vel_new, &pos_new, &ws_out, &mcl_out);

	// Start Timing 
	TIMING_INIT(totalTime);
	TIMING_START(totalTime);

	// Initialization
	initializeData(N, 0.9f, (F3)randn3, (F3)rand3, \
	               (F4)quat, (F3)vel, (F3)pos);
#ifdef CUDA
	CUDA_CHECK("initializeData");
	cudaThreadSynchronize();
#endif
	// End Timing
	TIMING_STOP(totalTime);

	// Main loop
	for (row = 0; row < in_rows; )
	{
		float data[8];
		int sType;
		readSensorData(data, &sType, inputData, in_cols, &row);
		float seed = RAND01();

		// Start Timing
		TIMING_START(totalTime);

		switch (sType)
		{
		case 1: entry_type1(N, seed, rand1, mcl_out, ws_out, \
		                    (F4)quat, (F3)pos, (F3)vel, \
		                    (F4)quat_new, (F3)pos_new, (F3)vel_new);
		        break;
		case 2: entry_type2(N, data[0], data[1], data[2], \
		                    (F4)quat, (F3)randn3);
		        break;
		case 3: entry_type3(N, data[0], data[1], data[2], seed, \
		                    (F3)randn3, rand1, mcl_out, ws_out, \
		                    (F4)quat, (F3)pos, (F3)vel, \
		                    (F4)quat_new, (F3)pos_new, (F3)vel_new);
		        break;
		case 4: entry_type4(N, data[0], data[1], data[2], \
		                    data[6], data[7], seed, \
		                    (F3)randn3, rand1, mcl_out, ws_out, \
		                    (F4)quat, (F3)pos, (F3)vel, \
		                    (F4)quat_new, (F3)pos_new, (F3)vel_new);
		        break;
		}
#ifdef CUDA
		cudaThreadSynchronize();
#endif
		if (sType != 2)
		{
			swap_ptrs(&quat, &quat_new);
			swap_ptrs(&pos, &pos_new);
			swap_ptrs(&vel, &vel_new);
		}

		// End Timing
		TIMING_STOP(totalTime);
#ifndef NOVERIFY
		// Verify results
		verifyResult(validFile, N, row, sType, quat, pos, vel);
#endif
	}

	// Print timing
	TIMING_PRINT(totalTime);

	// Clean up
	deallocateData(randn3, rand3, rand1, quat, vel, pos, \
	               quat_new, vel_new, pos_new, ws_out, mcl_out);

	fclose(validFile);
	free(inputData);

	return 0;
}
