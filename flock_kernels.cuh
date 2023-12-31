//-----------------------------------------------------------------------------
// Flock v2
// Copyright (C) 2023. Rama Hoetzlein
//-----------------------------------------------------------------------------

#ifndef DEF_FLOCK_CUH
	#define DEF_FLOCK_CUH

	#include <curand.h>
	#include <curand_kernel.h>
	#include <stdio.h>
	#include <math.h>

	#define CUDA_KERNEL
	#include "flock_types.h"

	#define EPSILON					0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			2147483647			// max int
	#define TOTAL_THREADS		1000000
	#define BLOCK_THREADS		256
	#define MAX_NBR					80		
	#define FCOLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	typedef unsigned int				uint;
	typedef unsigned short int	ushort;
	typedef unsigned char				uchar;
	
	extern "C" {
		__global__ void insertParticles ( int pnum );		
		__global__ void countingSortFull ( int pnum );				
		__global__ void advanceParticles ( float time, float dt, float ss, int numPnts );				
		__global__ void prefixFixup ( uint *input, uint *aux, int len);
		__global__ void prefixSum ( uint* input, uint* output, uint* aux, int len, int zeroff );		
	}

#endif
