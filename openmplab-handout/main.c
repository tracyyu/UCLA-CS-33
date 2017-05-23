#include "header.h"

////////////////////////////////////////////////////////////////////////////////

F2D*
readFile(const char* fileName)
{
	int n, rows, cols, i, j;
	FILE* fp = fopen(fileName, "r");
	if(!fp) return NULL;
	F2D* fill = NULL;

	n = fscanf(fp, "%d", &cols);
	if (n != 1) goto done;
	n = fscanf(fp, "%d", &rows);
	if (n != 1) goto done;

	fill = fSetArray(rows, cols, 0);

	for(i=0; i<rows; i++)
	for(j=0; j<cols; j++) {
		n = fscanf(fp, "%f", &(subsref(fill,i,j)));
		if (n != 1) { fFreeHandle(fill); fill = NULL; goto done; }
	}
done:
	fclose(fp);    
	return fill;
}

////////////////////////////////////////////////////////////////////////////////

F2D*
readSensorData(I2D* index, F2D* fid, I2D* type, I2D* eof)
{
    F2D *retData;
    int rows, i, k;
    int atype=-1, aindex;

    aindex = asubsref(index, 0);

    asubsref(index,0) = asubsref(index,0) + 1;
    rows = fid->height;
    asubsref(type,0) = 0;
    retData = fSetArray(1, 8, 0);
    
    if( asubsref(index,0) > (rows-1) )
        asubsref(eof,0) = 1;
    else
    {
        if( asubsref(index,0) == rows)
            asubsref(eof,0) = 1;
        else
            asubsref(eof,0) = 0;

        k = asubsref(index,0);
        atype = (int) subsref(fid, k, 1);
        if( (atype == 1) || (atype == 2) || (atype == 3) )
        {
            for(i=0; i<3; i++)
            {
                asubsref(retData,i) = subsref(fid,k,(i+2));
            }
        }
        if( atype == 4 )
        {
            for(i=0; i<3; i++)
            {
                asubsref(retData,i) = subsref(fid,k,(i+2));
            }
            for(i=3; i<8; i++)
            {
                asubsref(retData,i) = subsref(fid,k+1,(i-3));
            }
            aindex = aindex + 1;
        }
        aindex = aindex + 1;
    }

    asubsref(index,0) = aindex;
    asubsref(type, 0) = atype;

    return retData;
}

////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char* argv[])
{
	int i, j, rows, cols;
	F2D *pos, *vel, *eul1, *eul2, *quat, *ones, *randW;
	F2D *fid, *sData, *STDDEV_GPSPos;
	I2D *sType, *isEOF, *index;
	const int n = 2000;

	if(argc < 3) {
		fprintf(stderr, "Usage: %s INFILE VALIDFILE\n", argv[0]);
		return -1;
	}

	/** Open input files **/
	fid = readFile(argv[1]);
	if (!fid) {
		fprintf(stderr, "Error opening `%s`.\n", argv[1]);
		return -1;
	}
	FILE* validFile = fopen(argv[2],"r");
	if (!validFile) {
		fprintf(stderr, "Error opening `%s`.\n", argv[2]);
		return -1;
	}

#ifdef _OPENMP
	omp_set_num_threads(4);
#endif

	/** Start Timing **/ 
	TIMING_INIT(totalTime);
	TIMING_START(totalTime);

	/** Initialization **/
	pos = fSetArray(n, 3, 0);
	vel = fSetArray(n, 3, 0);
	ones = fSetArray(n,1,1);
	
	F2D *randn = randWrapper(n,3);
	for(i=0; i<n; i++)
	for(j=0; j<3; j++)
		subsref(vel,i,j) += subsref(randn,i,j) * STDDEV_ODOVel;
	fFreeHandle(randn);

	F2D *eulAngle;
	eulAngle = fSetArray(n, 3, 0);
	randn = randWrapper(n,1);

	for(i=0; i<n; i++)
		subsref(eulAngle, i, 2) = subsref(randn, i, 0) * 2 * M_PI;

	eul1 = eul2quat(eulAngle); 
	fFreeHandle(eulAngle);

	eulAngle = fSetArray(1, 3, 0);
	subsref(eulAngle, 0, 0) = M_PI;
	eul2 = eul2quat(eulAngle);

	fFreeHandle(randn);
	fFreeHandle(eulAngle);
    
	quat = quatMul(eul1, eul2);
	fFreeHandle(eul1);
	fFreeHandle(eul2);

	index = iSetArray(1,1,-1);
	sType = iSetArray(1,1,-1);
	isEOF = iSetArray(1,1,-1);

	rows = 0;
	cols = 5;
	STDDEV_GPSPos = fSetArray(3,3,0);
	randW = randnWrapper(n,3);

	/** End Timing **/   
	TIMING_STOP(totalTime);

	/** Main loop **/
	while (1)
	{
		sData = readSensorData(index, fid, sType, isEOF);    
		rows++;

		/** Start Timing **/ 
		TIMING_START(totalTime);

		switch (asubsref(sType,0))
		{
		case 2:
		entry_type2(sData, ones, &quat, randW);
		break;
		case 4:
		entry_type4(sData, ones, quat, pos, vel, STDDEV_GPSPos, randW);
		break;
		case 1:
		entry_type1(sData, quat, pos, vel);
		break;
		case 3:
		entry_type3(sData, ones, quat, &pos, &vel);
		break;
		}

		/** End Timing**/   
		TIMING_STOP(totalTime);

		fFreeHandle(sData);

		/** Verify results **/
		float quatOut=0, velOut=0, posOut=0;
		for(i=0; i<(quat->height*quat->width); i++)
			quatOut += asubsref(quat, i);
		for(i=0; i<(vel->height*vel->width); i++)
			velOut += asubsref(vel, i);
		for(i=0; i<(pos->height*pos->width); i++)
			posOut += asubsref(pos, i);

		union { float f; unsigned u; } valid[3];
		fscanf(validFile,"%X %X %X\n", &valid[0].u, &valid[1].u, &valid[2].u);
		if (fabsf((quatOut - valid[0].f)/valid[0].f) > 0.005f ||
		    fabsf((velOut - valid[1].f)/valid[1].f)  > 0.005f ||
		    fabsf((posOut - valid[2].f)/valid[2].f)  > 0.005f)
		{
			fprintf(stderr, "Invalid result at row %d!\n", rows);
			fprintf(stderr, "   Your values: %f %f %f\n", quatOut, velOut, posOut);
			fprintf(stderr, "Correct values: %f %f %f\n", valid[0].f, valid[1].f, valid[2].f);
			return -1;
		}

		/** Stop at end of file **/
		if (asubsref(isEOF,0) == 1)
			break;
	}

	/** Print timing **/
	TIMING_PRINT(totalTime);

#ifdef __APPLE__
	/** Mac OS X complains if there are no OpenMP pragmas */
	#pragma omp parallel
	{
		int x;
		x = 0;
	}
#endif

	/** Clean up **/
	fclose(validFile);
	fFreeHandle(STDDEV_GPSPos);
	iFreeHandle(index);
	iFreeHandle(sType);
	iFreeHandle(isEOF);
	fFreeHandle(fid);
	fFreeHandle(pos);
	fFreeHandle(vel);
	fFreeHandle(quat);
	fFreeHandle(ones);
	fFreeHandle(randW);
	
	return 0;
}
