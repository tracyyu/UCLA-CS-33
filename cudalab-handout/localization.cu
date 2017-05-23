#include "header.h"

////////////////////////////////////////////////////////////////////////////////

void
randWrapper(float out[], int rows, int cols, float seed)
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
randnWrapper(float out[], int rows, int cols, float seed)
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

void
mcl(float out[], int N, float rand1[], float seed, int n_channel)
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
weightedSample(int out[], int N, float rand1[], float seed, float w[])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
generateSample(int N, float seed, float w[], float rand1[], int sampleXId[], \
               float quat[][4], float vel[][3], float pos[][3], \
               float retQuat[][4], float retVel[][3], float retPos[][3])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type1(int N, float seed, float rand1[], float mcl_out[], int ws_out[], \
            float quat[][4], float pos[][3], float vel[][3], \
            float quat2[][4], float pos2[][3], float vel2[][3])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type2(int N, float x, float y, float z, \
            float quat[][4], float randn3[][3])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type3(int N, float acc0, float acc1, float acc2, float seed, \
            float randn3[][3], float rand1[], float mcl_out[], int ws_out[], \
            float quat[][4], float pos[][3], float vel[][3], \
            float quat2[][4], float pos2[][3], float vel2[][3])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type4(int N, float d0, float d1, float d2, \
            float d6, float d7, float seed, \
            float randn3[][3], float rand1[], float mcl_out[], int ws_out[], \
            float quat[][4], float pos[][3], float vel[][3], \
            float quat2[][4], float pos2[][3], float vel2[][3])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
allocateData(int N, float** randn3, float** rand3, float** rand1, \
             float** quat, float** vel, float** pos, \
             float** quat2, float** vel2, float** pos2, \
             int** ws_out, float** mcl_out)
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
initializeData(int N, float seed, float randn3[][3], float rand3[][3], \
               float quat[][4], float vel[][3], float pos[][3])
{
	// ...
}

////////////////////////////////////////////////////////////////////////////////

void
deallocateData(float* randn3, float* rand3, float* rand1, \
               float* quat , float* vel , float* pos, \
               float* quat2, float* vel2, float* pos2, \
               int* ws_out, float* mcl_out)
{
	// ...
}
