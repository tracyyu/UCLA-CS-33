#include "header.h"

void
randWrapper(float out[], int rows, int cols, float seed)
{
	int i,j;
	#pragma omp parallel for private(i,j)
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		float tmp = (i < j) ? (float)(i+1)/(float)(j+1) : (float)(j+1)/(float)(i+1);
		out[i*cols+j] = seed * tmp;
	}
}

void
randnWrapper(float out[], int rows, int cols, float seed)
{
	int i,j;
	#pragma omp parallel for private(i,j)
	for (i = 0; i < rows; i++)
	for (j = 0; j < cols; j++)
	{
		float tmp = (i < j) ? (float)(i+1)/(float)(j+1) : (float)(j+1)/(float)(i+1);
		float w = seed * tmp;
		float x = (-2.f * logf(w)) / w;
		out[i*cols+j] = x;
	}
}

// convert Euler angles to quaternion
void
euler2quat(float quat[4], float euler[3])
{
	float x = euler[0] * 0.5f;
	float y = euler[1] * 0.5f;
	float z = euler[2] * 0.5f;

	float Cx = cosf(x);
	float Cy = cosf(y);
	float Cz = cosf(z);
	float Sx = sinf(x);
	float Sy = sinf(y);
	float Sz = sinf(z);

	quat[0] = Cx*Cy*Cz + Sx*Sy*Sz;
	quat[1] = Sx*Cy*Cz - Cx*Sy*Sz;
	quat[2] = Cx*Sy*Cz + Sx*Cy*Sz;
	quat[3] = Cx*Cy*Sz - Sx*Sy*Cz;
}

// multiply quaternions
void
quat_mult(float out[4], float qa[4], float qb[4])
{
	float a0 = qa[0];
	float a1 = qa[1];
	float a2 = qa[2];
	float a3 = qa[3];

	float b0 = qb[0];
	float b1 = qb[1];
	float b2 = qb[2];
	float b3 = qb[3];

	out[0] = a0*b0 - a1*b1 - a2*b2 - a3*b3;
	out[1] = a0*b1 + a1*b0 + a2*b3 - a3*b2;
	out[2] = a0*b2 - a1*b3 + a2*b0 + a3*b1;
	out[3] = a0*b3 + a1*b2 - a2*b1 + a3*b0;
}

// rotate vector using quaternion
void
quat_rot(float out[3], float vec[3], float quat[4])
{
	float vx = vec[0];
	float vy = vec[1];
	float vz = vec[2];

	float q0 = quat[0];
	float q1 = quat[1];
	float q2 = quat[2];
	float q3 = quat[3];

	// P = Q * <0,V>
	float p0 = q0*0  - q1*vx - q2*vy - q3*vz;
	float p1 = q0*vx + q1*0  + q2*vz - q3*vy;
	float p2 = q0*vy - q1*vz + q2*0  + q3*vx;
	float p3 = q0*vz + q1*vy - q2*vx + q3*0 ;

	// R = P * Q'
	out[0] = -p0*q1 + p1*q0 - p2*q3 + p3*q2;
	out[1] = -p0*q2 + p1*q3 + p2*q0 - p3*q1;
	out[2] = -p0*q3 - p1*q2 + p2*q1 + p3*q0;
}

void
mcl(float out[], int N, float rand1[], float seed, int n_channel)
{
	int i;
	float scale = sqrtf(1.f / powf(2.f*M_PI, n_channel));
	float sum = 0.f;
	randWrapper(rand1,N,1,seed);

	#pragma omp parallel private(i)
	{
		#pragma omp for reduction(+:sum)
		for (i = 0; i < N; i++)
		{
			float f = expf(-0.5f * rand1[i]) * scale;
			out[i] = f;
			sum += f;
		}
		#pragma omp for
		for (i = 0; i < N; i++)
			out[i] /= sum;
	}
}

void
weightedSample(int out[], int N, float rand1[], float seed, float w[])
{
	int i, j;
	randWrapper(rand1,N,1,seed);

	#pragma omp parallel for private(i,j)
	for (j = 0; j < N; j++)
	{
		float rand_j = rand1[j];
		for (i = 0; i < N; i++)
		{
			if (rand_j <= 0) break;
			rand_j -= w[i];
		}
		out[j] = i;
	}
}

void
generateSample(int N, float seed, float w[], float rand1[], int sampleXId[], \
               float quat[][4], float vel[][3], float pos[][3], \
               float retQuat[][4], float retVel[][3], float retPos[][3])
{
	int i;
	weightedSample(sampleXId,N,rand1,seed,w);

	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++)
	{
		int index = sampleXId[i] - 1;

		retQuat[i][0] = quat[index][0];
		retQuat[i][1] = quat[index][1];
		retQuat[i][2] = quat[index][2];
		retQuat[i][3] = quat[index][3];

		retVel[i][0] = vel[index][0];
		retVel[i][1] = vel[index][1];
		retVel[i][2] = vel[index][2];

		retPos[i][0] = pos[index][0];
		retPos[i][1] = pos[index][1];
		retPos[i][2] = pos[index][2];
	}
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type1(int N, float seed, float rand1[], float mcl_out[], int ws_out[], \
            float quat[][4], float pos[][3], float vel[][3], \
            float quat2[][4], float pos2[][3], float vel2[][3])
{
	mcl(mcl_out, N, rand1, seed, 1);
	generateSample(N, seed, mcl_out, rand1, ws_out, quat, vel, pos, quat2, vel2, pos2);
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type2(int N, float x, float y, float z, \
            float quat[][4], float randn3[][3])
{
	int i;
	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++)
	{
		float g0 = x + randn3[i][0] * STDDEV_GYRO;
		float g1 = y + randn3[i][1] * STDDEV_GYRO;
		float g2 = z + randn3[i][2] * STDDEV_GYRO;

		float norm = sqrtf(g0*g0 + g1*g1 + g2*g2);

		g0 /= norm;
		g1 /= norm;
		g2 /= norm;

		float tmp = norm * GYRO_DT * 0.5f;
		float cosA = cosf(tmp);
		float sinA = sinf(tmp);

		float gquat[] = {cosA, sinA*g0, sinA*g1, sinA*g2};

		quat_mult(quat[i],quat[i],gquat);
	}
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type3(int N, float acc0, float acc1, float acc2, float seed, \
            float randn3[][3], float rand1[], float mcl_out[], int ws_out[], \
            float quat[][4], float pos[][3], float vel[][3], \
            float quat2[][4], float pos2[][3], float vel2[][3])
{
	int i;
	mcl(mcl_out, N, rand1, seed, 3);
	generateSample(N, seed, mcl_out, rand1, ws_out, quat, vel, pos, quat2, vel2, pos2);

	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++)
	{
		float qconj[] = { quat2[i][0], -quat2[i][1], -quat2[i][2], -quat2[i][3] };

		float s[3];
		quat_rot(s, vel2[i], qconj); // s = vel[i] ROT quat[i]'

		float g[] = {0.f,0.f,-9.8f};
		quat_rot(g, g, quat2[i]); // g = [0 0 -9.8] ROT quat[i]

		float a[] = {acc0-g[0], acc1-g[1], acc2-g[2]}; // a = acc - g

		// vel[i] += a * ACCL_DT  +  randn3[i] * STDDEV_VEL
		vel2[i][0] += a[0]*ACCL_DT + randn3[i][0]*STDDEV_VEL;
		vel2[i][1] += a[1]*ACCL_DT + randn3[i][1]*STDDEV_VEL;
		vel2[i][2] += a[2]*ACCL_DT + randn3[i][2]*STDDEV_VEL;

		// a = a ROT quat[i]'
		quat_rot(a, a, qconj);

		// pos[i] += s * ACCL_DT  +  0.5 * a * ACCL_DT^2
		pos2[i][0] += (s[0] + a[0]*0.5f*ACCL_DT)*ACCL_DT + randn3[i][0]*STDDEV_POS;
		pos2[i][1] += (s[1] + a[1]*0.5f*ACCL_DT)*ACCL_DT + randn3[i][1]*STDDEV_POS;
		pos2[i][2] += (s[2] + a[2]*0.5f*ACCL_DT)*ACCL_DT + randn3[i][2]*STDDEV_POS;
	}
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type4(int N, float d0, float d1, float d2, \
            float d6, float d7, float seed, \
            float randn3[][3], float rand1[], float mcl_out[], int ws_out[], \
            float quat[][4], float pos[][3], float vel[][3], \
            float quat2[][4], float pos2[][3], float vel2[][3])
{
	int i;
	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++)
	{
		pos[i][0] =   d6 * randn3[i][0] + d0;
		pos[i][1] =   d7 * randn3[i][1] + d1;
		pos[i][2] = 15.f * randn3[i][2] + d2;
	}

	mcl(mcl_out, N, rand1, seed, 4);
	generateSample(N, seed, mcl_out, rand1, ws_out, quat, vel, pos, quat2, vel2, pos2);
}

////////////////////////////////////////////////////////////////////////////////

#define ALLOC_FLOAT(h,w) \
	((float*)malloc((h)*(w)*sizeof(float)))

#define ALLOC_INT(h,w) \
	((int*)malloc((h)*(w)*sizeof(int)))

void
allocateData(int N, float** randn3, float** rand3, float** rand1, \
             float** quat, float** vel, float** pos, \
             float** quat2, float** vel2, float** pos2, \
             int** ws_out, float** mcl_out)
{
	*randn3 = ALLOC_FLOAT(N,3);
	*rand3  = ALLOC_FLOAT(N,3);
	*rand1  = ALLOC_FLOAT(N,1);

	*quat  = ALLOC_FLOAT(N,4);
	*quat2 = ALLOC_FLOAT(N,4);
	*vel   = ALLOC_FLOAT(N,3);
	*vel2  = ALLOC_FLOAT(N,3);
	*pos   = ALLOC_FLOAT(N,3);
	*pos2  = ALLOC_FLOAT(N,3);

	*ws_out = ALLOC_INT(N,1);
	*mcl_out = ALLOC_FLOAT(N,1);
}

void
initializeData(int N, float seed, float randn3[][3], float rand3[][3], \
               float quat[][4], float vel[][3], float pos[][3])
{
	int i;
	float quat0[4];
	float angle0[] = {M_PI,0,0};
	euler2quat(quat0,angle0);

	// fill pos with zeros
	memset(pos, 0, N*3*sizeof(float));

	// fill random values
	randnWrapper((float*)randn3,N,3,seed);
	randWrapper((float*)rand3,N,3,seed);

	#pragma omp parallel for private(i)
	for (i = 0; i < N; i++)
	{
		// set velocity
		vel[i][0] = rand3[i][0] * STDDEV_ODOVel;
		vel[i][1] = rand3[i][1] * STDDEV_ODOVel;
		vel[i][2] = rand3[i][2] * STDDEV_ODOVel;

		// set quaternion
		float q[4];
		float a[] = { 0, 0, rand3[i][0]*(2.f*M_PI) };
		euler2quat(q,a);
		quat_mult(quat[i], q, quat0);
	}
}

void
deallocateData(float* randn3, float* rand3, float* rand1, \
               float* quat , float* vel , float* pos, \
               float* quat2, float* vel2, float* pos2, \
               int* ws_out, float* mcl_out)
{
	free(randn3);
	free(rand3);
	free(rand1);
	free(quat);
	free(quat2);
	free(vel);
	free(vel2);
	free(pos);
	free(pos2);
	free(ws_out);
	free(mcl_out);
}
