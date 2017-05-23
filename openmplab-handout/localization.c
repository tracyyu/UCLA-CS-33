#include "header.h"

////////////////////////////////////////////////////////////////////////////////

I2D*
iMallocHandle(int rows, int cols)
{
    I2D* out;
    out = (I2D*)malloc(sizeof(I2D) + sizeof(int)*rows*cols);
    out->height = rows;
    out->width = cols;
    return out;
}

void
iFreeHandle(I2D* out)
{
    if(out != NULL)
        free(out);
}

I2D*
iSetArray(int rows, int cols, int val)
{
    int i, j;
    I2D *out = iMallocHandle(rows, cols);

#pragma omp parallel for private(j) 
    for(i=0; i<rows; i++)
    for(j=0; j<cols; j++)
        subsref(out,i,j) = val;

    return out;
}

////////////////////////////////////////////////////////////////////////////////

F2D*
fMallocHandle(int rows, int cols)
{
    F2D* out;
    out = (F2D*)malloc(sizeof(F2D) + sizeof(float)*rows*cols);
    out->height = rows;
    out->width = cols;
    return out;
}

void
fFreeHandle(F2D* out)
{
    if(out != NULL)
        free(out);
}

F2D*
fSetArray(int rows, int cols, float val)
{
    int i, j;
    F2D *out;
    out = fMallocHandle(rows, cols);
#pragma omp parallel for private(j) 
for(i=0; i<rows; i++){
    for(j=0; j<cols; j++)
        subsref(out,i,j) = val;
}
    return out;
    
}

F2D*
fDeepCopy(F2D* in)
{
    int i, j;
    F2D* out;
    int rows, cols;
    
    rows = in->height;
    cols = in->width;

    out = fMallocHandle(rows, cols);
    
    for(i=0; i<rows; i++)
    for(j=0; j<cols; j++)
        subsref(out,i,j) = subsref(in,i,j);

    return out;
}

F2D*
fDeepCopyRange(F2D* in, int startRow, int numRows, int startCol, int numCols)
{
    int i, j, k;
    F2D *out;
    int rows, cols;
    
    rows = numRows + startRow;
    cols = numCols + startCol;
    out = fMallocHandle(numRows, numCols);
    
    k = 0;
    for(i=startRow; i<rows; i++)
        for(j=startCol; j<cols; j++)
            asubsref(out,k++) = subsref(in,i,j);
    
    return out;
}

F2D*
fHorzcat(F2D* a, F2D* b)
{
    F2D *out;
    int rows=0, cols=0, i, j, k, c_1, c_2;
    int r_1;

    r_1 = a->height;
    c_1 = a->width;
    cols += c_1;
    c_2 = b->width;
    cols += c_2;
    rows = r_1;

    out = fMallocHandle(rows, cols);    
    
    for(i=0; i<rows; i++)
    {
        k = 0;
        for(j=0; j<c_1; j++)
        {
            subsref(out,i,k) = subsref(a,i,j);
            k++;
        }
        for(j=0; j<c_2; j++)
        {
            subsref(out,i,k) = subsref(b,i,j);
            k++;
        }
    }

    return out;
}

F2D*
fPlus(F2D* a, F2D* b)
{
    F2D *c;
    int i, rows, cols;

    rows = a->height;
    cols = a->width;

    c = fMallocHandle(rows, cols);
    
//#pragma omp parallel for 
    for(i=0; i<(rows*cols); i++)
        asubsref(c,i) = asubsref(a,i) + asubsref(b,i);

    return c;
}

F2D*
fMinus(F2D* a, F2D* b)
{
    F2D *c;
    int i, rows, cols;

    rows = a->height;
    cols = a->width;

    c = fMallocHandle(rows, cols);
//#pragma omp parallel for     
    for(i=0; i<(rows*cols); i++)
        asubsref(c,i) = asubsref(a,i) - asubsref(b,i);

    return c;
}

F2D*
fSum(F2D* inMat)
{
    F2D *outMat;
    int rows, cols, i, j;
    float temp;
    int Rcols;

    rows = inMat->height;
    cols = inMat->width;

//#pragma omp parallel
    if(cols == 1 || rows == 1)
        Rcols = 1;
    else
        Rcols = cols;

    outMat = fSetArray(1,Rcols,0);

    if( cols == 1)
    {    
        temp = 0;
        for( j=0; j<rows; j++)
            temp = temp + subsref(inMat,j,0);
        asubsref(outMat,0) = temp;
    }
    else if( rows == 1)
    {
        temp = 0;
        for( j=0; j<cols; j++)
            temp = temp + asubsref(inMat,j);
        asubsref(outMat,0) = temp;
    }
    else
    {
        for( i=0; i<cols; i++)
        {
            temp = 0;
            for( j=0; j<rows; j++)
                temp = temp + subsref(inMat,j,i);
            asubsref(outMat,i) = temp;
        }
    }

    return outMat;
}

F2D*
fSum2(F2D* inMat, int dir)
{
    F2D *outMat;
    int rows, cols, i, j;
    float temp;
    int newRow, newCols;

    rows = inMat->height;
    cols = inMat->width;

    if(dir == 1)
    { 
        newRow = 1;
        newCols = cols;
    }
    else
    {
        newRow = rows;
        newCols = 1;
    }

    outMat = fSetArray(newRow,newCols,0);

    if(dir == 1)
    {
        for (i=0; i<cols; i++)
        {
            temp = 0;
            for( j=0; j<rows; j++)
                temp = temp + subsref(inMat,j,i);
            asubsref(outMat,i) = temp;
        }
    }
    else
    {
        for( i=0; i<rows; i++)
        {
            temp = 0;
            for( j=0; j<cols; j++)
                temp = temp + subsref(inMat,i,j);
            subsref(outMat,i,0) = temp;
        }
    }

    return outMat;
}

F2D*
fTimes(F2D* a, F2D* b)
{
    F2D *c;
    int i, rows, cols;

    rows = a->height;
    cols = a->width;

    c = fMallocHandle(rows, cols);
    
    for(i=0; i<(rows*cols); i++)
        asubsref(c,i) = asubsref(a,i) * asubsref(b,i);

    return c;
}

F2D*
fDivide(F2D* a, float b)
{
    F2D *c;
    int i, rows, cols;

    rows = a->height;
    cols = a->width;

    c = fMallocHandle(rows, cols);
   
    for(i=0; i<(rows*cols); i++)
    {
        asubsref(c,i) = asubsref(a,i) / b;
    }

    return c;
}

F2D*
ffDivide(F2D* a, F2D* b)
{
    F2D *c;
    int i, rows, cols;

    rows = a->height;
    cols = a->width;

    c = fMallocHandle(rows, cols);
    
    for(i=0; i<(rows*cols); i++)
        asubsref(c,i) = asubsref(a,i) / asubsref(b,i);

    return c;
}

F2D*
fMtimes(F2D* a, F2D* b)
{
    F2D *out;
    int m, p, p1, n, i, j, k;
    float temp;

    m = a->height;
    p = a->width;

    p1 = b->height;
    n = b->width;

    out = fMallocHandle(m,n);

    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            temp = 0;
            for(k=0; k<p; k++)
            {
                temp += subsref(b,k,j) * subsref(a,i,k);
            }
            subsref(out,i,j) = temp;
        }
    }

    return out;
}

F2D*
fMdivide(F2D* a, F2D* b)
{
    F2D *c;
    int i, rows, cols;

    rows = a->height;
    cols = a->width;

    if(rows != b->height || cols != b->width)
    {
        printf("fMDivide Mismatch = \nrows: %d\t%d\ncols: %d\t%d\n", \
		rows, b->height, cols, b->width);
        return NULL;
    }

    c = fMallocHandle(rows, cols);
//#pragma omp parallel    
    for(i=0; i<(rows*cols); i++)
        asubsref(c,i) = asubsref(a,i) / asubsref(b,i);

    return c;
}

////////////////////////////////////////////////////////////////////////////////

F2D*
randWrapper(int m, int n)
{
    F2D *out;
    float seed;
    int i,j;

    out = fSetArray(m, n, 0);
    seed = 0.9;

#pragma omp parallel for private(j)
    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            if(i<j)
                subsref(out,i,j) = seed * ((i+1.0)/(j+1.0));
            else
                subsref(out,i,j) = seed * ((j+1.0)/(i+1.0));
        }
    }

    return out;
}

F2D*
randnWrapper(int m, int n)
{
    F2D *out;
    float seed;
    int i,j;

    out = fSetArray(m, n, 0);
    seed = 0.9;

#pragma omp parallel for private(j)
    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            if(i<j)
                subsref(out,i,j) = seed * ((i+1.0)/(j+1.0));
            else
                subsref(out,i,j) = seed * ((j+1.0)/(i+1.0));
        }
    }
#pragma omp parallel for private(j)
    for(i=0; i<m ;i++)
    {
        for(j=0; j<n; j++)
        {
            float w;
            w = subsref(out,i,j);
            w = ((-2.0 * log(w))/w);
            subsref(out,i,j) = w;
        }
    }

    return out;
}

////////////////////////////////////////////////////////////////////////////////

F2D*
quat2eul(F2D* quat)
{
    F2D *retEul;
    int i, k = 0;
    int rows, cols;

    rows = quat->height;
    cols = quat->width;

    retEul = fSetArray(rows, 3, 0);

#pragma omp parallel for
    for(i=0; i<rows; i++)
    {
        float temp, temp1, temp2, temp3, temp4;
        float quati2, quati3, quati1, quati0;

        quati0 = subsref(quat,i,0);
        quati1 = subsref(quat,i,1);
        quati2 = subsref(quat,i,2);
        quati3 = subsref(quat,i,3);

        temp = 2 *quati2 * quati3 + quati0 * quati1;
        temp1 = pow(quati0,2) - pow(quati1,2) - pow(quati2,2) + pow(quati3,2);
        temp2 = -2*quati1 * quati2 + quati0 * quati3;
        temp3 = 2*quati1 * quati2 + quati0 * quati3;
        temp4 = pow(quati0,2) + pow(quati1,2) - pow(quati2,2) - pow(quati3,2);
        
        asubsref(retEul,k++) = atan2(temp, temp1);
        asubsref(retEul,k++) = asin(temp2);
        asubsref(retEul,k++) = atan2(temp3, temp4);
    }

    return retEul;
}

F2D*
eul2quat(F2D* angle)
{
    F2D *ret;
    F2D *x, *y, *z;
    int k, i;
    int rows, cols;

    rows = angle->height;
    cols = angle->width;

    x = fDeepCopyRange(angle, 0, angle->height, 0, 1);
    y = fDeepCopyRange(angle, 0, angle->height, 1, 1);
    z = fDeepCopyRange(angle, 0, angle->height, 2, 1);

    ret = fSetArray(x->height, 4, 0);
    
    for(i=0; i<rows; i++)
    {
        float xi, yi, zi;
        k = 0;
        xi = asubsref(x,i);
        yi = asubsref(y,i);
        zi = asubsref(z,i);

        subsref(ret,i,k) = cos(xi/2)*cos(yi/2)*cos(zi/2)+sin(xi/2)*sin(yi/2)*sin(zi/2);
        k++;
        subsref(ret,i,k) = sin(xi/2)*cos(yi/2)*cos(zi/2)-cos(xi/2)*sin(yi/2)*sin(zi/2);
        k++;
        subsref(ret,i,k) = cos(xi/2)*sin(yi/2)*cos(zi/2)+sin(xi/2)*cos(yi/2)*sin(zi/2);
        k++;
        subsref(ret,i,k) = cos(xi/2)*cos(yi/2)*sin(zi/2)-sin(xi/2)*sin(yi/2)*cos(zi/2);
    }

    fFreeHandle(x);
    fFreeHandle(y);
    fFreeHandle(z);

    return ret;
}

F2D*
quatConj(F2D* a)
{
    F2D* retQuat;
    int rows, cols;
    int i, k;

    rows = a->height;
    cols = a->width;
    retQuat = fSetArray(rows, 4, 0);

//#pragma omp parallel for private(k)
    for(i=0; i<rows; i++)
    {
        k=0;
        subsref(retQuat,i,k++) = subsref(a,i,0);
        subsref(retQuat,i,k++) = -subsref(a,i,1);
        subsref(retQuat,i,k++) = -subsref(a,i,2);
        subsref(retQuat,i,k) = -subsref(a,i,3);
    }

    return retQuat;
}

F2D*
quatMul(F2D* a, F2D* b)
{
    int ra, ca, rb, cb;
    F2D *ret;
    int i, j, k=0;

    ra = a->height;
    ca = a->width;

    rb = b->height;
    cb = b->width;

    ret = fSetArray(ra, 4, 0);

    j = 0;
    for(i=0; i<ra; i++)
    {
        k = 0;
        float ai0, ai1, ai2, ai3;
        float bj0, bj1, bj2, bj3;

        ai0 = subsref(a,i,0);
        ai1 = subsref(a,i,1);
        ai2 = subsref(a,i,2);
        ai3 = subsref(a,i,3);
        
        bj0 = subsref(b,j,0);
        bj1 = subsref(b,j,1);
        bj2 = subsref(b,j,2);
        bj3 = subsref(b,j,3);
        
        subsref(ret,i,k++) = ai0*bj0 - ai1*bj1 - ai2*bj2 - ai3*bj3;
        subsref(ret,i,k++) = ai0*bj1 + ai1*bj0 + ai2*bj3 - ai3*bj2;
        subsref(ret,i,k++) = ai0*bj2 - ai1*bj3 + ai2*bj0 + ai3*bj1;
        subsref(ret,i,k++) = ai0*bj3 + ai1*bj2 - ai2*bj1 + ai3*bj0;
    
        if(rb == ra)
            j++;
    }

    return ret;
}

F2D*
quatRot(F2D* vec, F2D* rQuat)
{
    F2D *ret;
    int nr, i, j, k, rows, cols;
    F2D *tv, *vQuat, *temp, *temp1;
    F2D *retVec;

    nr = vec->height;
    tv = fSetArray(nr, 1, 0);
    vQuat = fHorzcat(tv, vec);
    temp = quatMul(rQuat, vQuat);
    temp1 = quatConj(rQuat);
    retVec = quatMul(temp, temp1);

    rows = retVec->height;
    cols = retVec->width;

    ret = fSetArray(rows, 3, 0);

    for(i=0; i<rows; i++)
    {
        k = 0;
        for(j=1; j<4; j++)
        {
            subsref(ret,i,k) = subsref(retVec,i,j);
            k++;
        }
    }

    fFreeHandle(tv);
    fFreeHandle(vQuat);
    fFreeHandle(temp);
    fFreeHandle(temp1);
    fFreeHandle(retVec);
        
    return ret;
}

////////////////////////////////////////////////////////////////////////////////

F2D*
get3DGaussianProb( F2D* data, F2D* mean, F2D* A)
{
    F2D *p, *diff, *temp2, *mt;
    float temp;
    int n_data, n_channel;
    int i, j;
    F2D* t;
    float pi = 3.1412;

    n_data = data->height;
    n_channel = data->width;

    t = fSetArray(n_data, 1, 1);

    mt = fMtimes(t, mean); 
    diff = fMinus( data, mt);
    p = fSetArray(diff->height, 1, 0);

    temp = sqrt(1.0/(pow(2*pi, n_channel)));
    temp2 = randWrapper(diff->height,1);

    j = (temp2->height*temp2->width);
  
for(i=0; i<j; i++)
    {
        float temp2i;
        temp2i = asubsref(temp2,i);

        temp2i = exp(-0.5*temp2i);
        asubsref(p,i) = temp2i*temp;
    }

    fFreeHandle(t);
    fFreeHandle(temp2);
    fFreeHandle(mt);
    fFreeHandle(diff);
    
    return p;
}

F2D*
mcl(F2D* x, F2D* sData, F2D* invConv)
{
    int i, j;
    F2D *retW, *retX, *sum;
    float sumVal;

    retX = fDeepCopy(x);
    retW = get3DGaussianProb(retX, sData, invConv);
    sum = fSum(retW);
    if(sum->height == 1 && sum->width ==1)
    {
        sumVal = asubsref(sum,0);
     
//#pragma omp parallel private(i,j)
//{
//#pragma omp for nowait
   for(i=0; i<retW->height; i++)
            for(j=0; j<retW->width; j++)
                subsref(retW,i,j) = subsref(retW,i,j)/sumVal;
    }
//}
    else
        retW = fMdivide(retW, sum);

    fFreeHandle(retX);
    fFreeHandle(sum);

    return retW;
}

////////////////////////////////////////////////////////////////////////////////

I2D*
weightedSample(F2D* w)
{
    I2D *bin;
    F2D *seed;
    int n, i, j;

    n = w->height;
    seed = randWrapper(n, 1);
    bin = iSetArray(n, 1, 0);
//omp_set_nested(1);

#pragma omp parallel private(i,j)
{
    for(i=0; i<n; i++)
    {
	#pragma omp for nowait
        for(j=0; j<n; j++)
        {
            if(asubsref(seed,j) > 0)
                asubsref(bin,j)++;
	    asubsref(seed,j) -= asubsref(w,i);
        }

/*#pragma omp for nowait
        for(j=0; j<n; j++)
            asubsref(seed,j) = asubsref(seed,j) - asubsref(w,i);
  */  }
}
    free(seed);
    return bin;
}

void
generateSample(F2D *w, F2D *quat, F2D *vel, F2D *pos)
{
    int rows, cols, i, j, index;
    I2D *sampleXId;
    F2D *retQuat, *retVel, *retPos;

    sampleXId = weightedSample(w);

    rows = sampleXId->height;
    cols = sampleXId->width;

    if(cols > 1) {
        printf("ERROR: Cols more than 1\n");
	return ;
    }

    retQuat = fSetArray(quat->height, quat->width, 0);
    retVel = fSetArray(vel->height, vel->width, 0);
    retPos = fSetArray(pos->height, pos->width, 0);

for(i=0; i<rows; i++)
    {
        index = asubsref(sampleXId, i) - 1;
        for(j=0; j<quat->width; j++)
        {
            subsref(retQuat,i,j) = subsref(quat,index,j);
        }
    }
    for(i=0; i<rows; i++)
    {
        index = asubsref(sampleXId, i) - 1;
        for(j=0; j<vel->width; j++)
        {
            subsref(retVel,i,j) = subsref(vel,index,j);
        }
    }
    for(i=0; i<rows; i++)
    {
        index = asubsref(sampleXId, i) - 1;
        for(j=0; j<pos->width; j++)
        {
            subsref(retPos,i,j) = subsref(pos,index,j);
        }
    }
    for(i=0; i<quat->height; i++)
    {
        for(j=0; j<quat->width; j++)
        {
            subsref(quat,i,j) = subsref(retQuat,i,j);
        }
    } 
    for(i=0; i<vel->height; i++)
    {
        for(j=0; j<vel->width; j++)
        {
            subsref(vel,i,j) = subsref(retVel,i,j);
        }
    }   
    for(i=0; i<pos->height; i++)
    {
        for(j=0; j<pos->width; j++)
        {
            subsref(pos,i,j) = subsref(retPos,i,j);
        }
    }
  
    fFreeHandle(retQuat);
    fFreeHandle(retVel);
    fFreeHandle(retPos);
    iFreeHandle(sampleXId);
    
    return;
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type2(F2D *sData, F2D *ones, F2D **quat, F2D *randW)
{
	// Motion model
	int i;
	F2D *t, *t1;
	F2D *abc, *abcd;
	int qD_r=0, qD_c=0;
	F2D *cosA, *sinA;
	int n = ones->height;

	t = fDeepCopyRange(sData, 0, 1, 0, 3);
	F2D *gyro = fMtimes(ones, t);
	abc = fMallocHandle(gyro->height, gyro->width);
	t1 = fDeepCopy(randW);

	for(i=0; i<(n*3); i++)
	{
	    asubsref(t1,i) = asubsref(randW,i) * M_STDDEV_GYRO;
	    asubsref(gyro, i) += asubsref(t1,i);
	    asubsref(abc, i) = pow(asubsref(gyro, i), 2);
	}
	fFreeHandle(t1);
	abcd = fSum2(abc, 2);

	F2D *norm_gyro = fMallocHandle(abcd->height,abcd->width);
	F2D *angleAlpha = fMallocHandle(abcd->height, abcd->width);
#pragma omp parallel for
	for(i=0; i<(abcd->height*abcd->width); i++)
	{
	    asubsref(norm_gyro, i) = sqrt(asubsref(abcd,i));
	    asubsref(angleAlpha,i) = asubsref(norm_gyro,i) * gyroTimeInterval;
	}

	qD_r += angleAlpha->height + gyro->height;
	qD_c += angleAlpha->width + 3;
	
	fFreeHandle(t);
	fFreeHandle(abcd); 
	
	cosA = fSetArray(angleAlpha->height, angleAlpha->width, 0);
	sinA = fSetArray(angleAlpha->height, angleAlpha->width, 0);

#pragma omp parallel for
	for(i=0; i<(cosA->height*cosA->width); i++)
	    asubsref(cosA,i) = cos( asubsref(angleAlpha,i) /2 );
#pragma omp parallel for			
	for(i=0; i<(sinA->height*sinA->width); i++)
	    asubsref(sinA,i) = sin( asubsref(angleAlpha,i) /2 );

	fFreeHandle(abc);
	abc = fSetArray(1,3,1);
	t1 = fMtimes(norm_gyro, abc);
	t = ffDivide(gyro, t1);
	fFreeHandle(t1);

	abcd = fMtimes(sinA, abc);
	t1 = fTimes(t, abcd);
	F2D *quatDelta = fHorzcat(cosA, t1);
	
	fFreeHandle(abcd);
	fFreeHandle(t);
	fFreeHandle(t1);
	fFreeHandle(abc);

	t = quatMul(*quat, quatDelta);
	fFreeHandle(*quat);
	fFreeHandle(quatDelta);
	*quat = fDeepCopy(t);
	
	fFreeHandle(t);
	fFreeHandle(norm_gyro);
	fFreeHandle(gyro);
	fFreeHandle(angleAlpha);
	fFreeHandle(cosA);
	fFreeHandle(sinA);
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type4(F2D *sData, F2D *ones, F2D *quat, F2D *pos, F2D *vel,
            F2D *STDDEV_GPSPos, F2D *randW)
{
	//Observation
	float tempSum=0;
	F2D *Ovel;
	float OvelNorm;
	int i;

	asubsref(STDDEV_GPSPos, 0) = asubsref(sData, 6);
	asubsref(STDDEV_GPSPos, 4) = asubsref(sData,7);
	asubsref(STDDEV_GPSPos, 8) = 15;

	F2D *Opos = fDeepCopyRange(sData, 0, 1, 0, 3);

	//Initialize
	for(i=0; i<(pos->height*pos->width); i++)
		tempSum += asubsref(pos,i);

	if(tempSum == 0)
	{
		F2D *t, *t1;
		t = fMtimes(randW, STDDEV_GPSPos);
		t1 = fMtimes(ones, Opos);

		for(i=0; i<(pos->height*pos->width); i++)
			asubsref(pos,i) = asubsref(t,i) + asubsref(t1,i);

		fFreeHandle(t);
		fFreeHandle(t1);
	}
	else 
	{
		int rows, cols;
		int mnrows, mncols;

		rows = STDDEV_GPSPos->height;
		cols = STDDEV_GPSPos->width;

		F2D* temp_STDDEV_GPSPos = fSetArray(rows,cols,1);
		for(mnrows=0; mnrows<rows; mnrows++)
		for(mncols=0; mncols<cols; mncols++)
		{
			subsref(temp_STDDEV_GPSPos,mnrows,mncols) = \
			    pow(subsref(STDDEV_GPSPos,mnrows,mncols),-1);
		}

		F2D *w = mcl(pos, Opos , temp_STDDEV_GPSPos);
		generateSample(w, quat, vel, pos);
		fFreeHandle(temp_STDDEV_GPSPos);
		fFreeHandle(w);
	}
	fFreeHandle(Opos);

	//compare direction
	Ovel = fDeepCopyRange(sData, 0, 1, 3, 3);
	OvelNorm=2;

	if (OvelNorm>0.5)
	{
		F2D *t;
		t = fDeepCopy(Ovel);
		fFreeHandle(Ovel);
		/* This is a double precision division */
		Ovel = fDivide(t, OvelNorm);
		F2D *qConj = quatConj(quat);
		fFreeHandle(t);

		t = fSetArray(1,3,0);
		subsref(t,0,0) = 1;
		F2D *orgWorld = quatRot(t, qConj);
		fFreeHandle(t);
		fFreeHandle(qConj);
		t = fSetArray(3,3,0);
		asubsref(t,0) = 1;
		asubsref(t,4) = 1;
		asubsref(t,8) = 1;

		int i;
//#pragma omp parallel for
		for(i=0; i<(t->height*t->width); i++)
			asubsref(t, i) = asubsref(t,i)/STDDEV_GPSVel;
		F2D *w = mcl( orgWorld, Ovel, t);
		generateSample(w, quat, vel, pos);

		fFreeHandle(t);
		fFreeHandle(w);
		fFreeHandle(orgWorld);
	}

	fFreeHandle(Ovel);
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type1(F2D *sData, F2D *quat, F2D *pos, F2D *vel)
{
	//Observation
	F2D *Ovel;
	F2D *t, *t1;
	float valVel;
	int i;

	t = fSetArray(vel->height, 1, 0);
//#pragma omp parallel for
	for(i=0; i<vel->height; i++)
	{
		subsref(t,i,0) = sqrt( pow(subsref(vel,i,0),2) + \
		                 pow(subsref(vel,i,1),2) + \
		                 pow(subsref(vel,i,2),2) );
	}

	Ovel = fSetArray(1, 1, asubsref(sData,0));
	valVel = 1.0/STDDEV_ODOVel;

	t1 = fSetArray(1,1,(1.0/STDDEV_ODOVel));
	F2D *w = mcl (t, Ovel, t1);
	generateSample(w, quat, vel, pos);

	fFreeHandle(w);
	fFreeHandle(t);
	fFreeHandle(t1);
	fFreeHandle(Ovel);
}

////////////////////////////////////////////////////////////////////////////////

void
entry_type3(F2D *sData, F2D *ones, F2D *quat, F2D **pos, F2D **vel)
{
	//Observation
	F2D *t;
	t = fSetArray(1, 3, 0);
	asubsref(t,2) = -9.8;

	F2D *accl = fDeepCopyRange(sData, 0, 1, 0, 3);
	F2D *gtemp = fMtimes( ones, t); 
	F2D *gravity = quatRot(gtemp, quat);

	fFreeHandle(gtemp);
	fFreeHandle(t);
	t = fSetArray(3,3,0);
	asubsref(t,0) = 1;
	asubsref(t,4) = 1;
	asubsref(t,8) = 1;

	int n = ones->height;
	int i;
	for(i=0; i<(t->height*t->width); i++)
		asubsref(t,i) = asubsref(t,i)/STDDEV_ACCL;
	F2D *w = mcl( gravity, accl, t);

	generateSample(w, quat, *vel, *pos);
	fFreeHandle(t);
	
	//Motion model
	t = fMtimes(ones, accl);
	fFreeHandle(accl);
	accl = fMinus(t, gravity);
	fFreeHandle(w);
	fFreeHandle(gravity);
	fFreeHandle(t);

	F2D *s, *is;
	is = quatConj(quat);
	s = quatRot(*vel, is);
	fFreeHandle(is);
#pragma omp parallel for
	for(i=0; i<(s->height*s->width); i++)
	{
		asubsref(s,i) = asubsref(s,i)*acclTimeInterval;
	}
	is = fPlus(*pos, s);
	fFreeHandle(*pos);
	*pos = fDeepCopy(is);
	fFreeHandle(is);
	fFreeHandle(s);

	/** pos_ above stores: pos+quatRot(vel,quatConj(quat))*acclTimeInterval **/

	is = quatConj(quat);
	s = quatRot(accl, is);
	t = fDeepCopy(s);

	for(i=0; i<(s->height*s->width); i++)
	{
		asubsref(t,i) = 1/2*asubsref(s,i)*acclTimeInterval*acclTimeInterval;
	}

	/** t_ above stores: 1/2*quatRot(accl,quatCong(quat))*acclTimeInterval^2 **/

	fFreeHandle(s);
	fFreeHandle(is);
	s = randnWrapper(n,3);
//#pragma omp parallel for
	for(i=0; i<(s->height*s->width); i++)
	{
		asubsref(s,i) = asubsref(s,i) * M_STDDEV_POS;
	}

	/** s_ above stores: randn(n,3)*M_STDDEV_POS **/

	is = fPlus(*pos, t);
	fFreeHandle(*pos);
	*pos = fPlus(is, s);

	fFreeHandle(s);
	fFreeHandle(t);
	fFreeHandle(is);

	//vel=vel+accl*acclTimeInterval+randn(n,3)*M_STDDEV_VEL;

	t = fDeepCopy(accl);
#pragma omp parallel for
	for(i=0; i<(accl->height*accl->width); i++)
	{
		asubsref(t,i) = asubsref(accl,i) * acclTimeInterval;
	}

	is = fPlus(*vel, t);
	fFreeHandle(accl);
	fFreeHandle(t);
	s = randnWrapper(n,3);

#pragma omp parallel for
	for(i=0; i<(s->height*s->width); i++)
	{
		asubsref(s,i) = asubsref(s,i) * M_STDDEV_VEL;
	}

	fFreeHandle(*vel);
	*vel = fPlus(is, s);
	fFreeHandle(is);
	fFreeHandle(s);
}
