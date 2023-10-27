//-----------------------------------------------------------------------------
// Flock v2
// Copyright (C) 2023. Rama Hoetzlein
//-----------------------------------------------------------------------------

#define CUDA_KERNEL

#include "flock_kernels.cuh"

#include "quaternion.cuh"

#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

#include "datax.h"

__constant__ Accel		FAccel;
__constant__ Params		FParams;
__constant__ cuDataX	FBirds;
__constant__ cuDataX	FBirdsTmp;
__constant__ cuDataX	FGrid;

#define SCAN_BLOCKSIZE		512

extern "C" __global__ void insertParticles ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	
	register float3	p;
	register float3	gcf;
	register int3		gc;	
	register int		gs;	

	p = ((Bird*) FBirds.data(FBIRD))[i].pos;
	gcf = (p - FAccel.gridMin) * FAccel.gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * FAccel.gridRes.z + gc.z) * FAccel.gridRes.x + gc.x;

	if ( gc.x >= 1 && gc.x <= FAccel.gridScanMax.x && gc.y >= 1 && gc.y <= FAccel.gridScanMax.y && gc.z >= 1 && gc.z <= FAccel.gridScanMax.z ) {
		FBirds.bufI(FGCELL)[i] = gs;													// Grid cell insert.
		FBirds.bufI(FGNDX)[i] = atomicAdd ( &FGrid.bufI(AGRIDCNT)[ gs ], 1 );		// Grid counts.
	} else {
		FBirds.bufI(FGCELL)[i] = GRID_UNDEF;		
	}	

	//--- debugging
	/*if ( i==0 ) {
		printf ( "FPnts:FPOINT: %012llx\n", FBirds.bufF3(FPOINT) );
		printf ( "gridRes: %d, %d, %d\n", FParams.gridRes.x, FParams.gridRes.y, FParams.gridRes.z );
		printf ( "gridScanMax: %d, %d, %d\n", FParams.gridScanMax.x, FParams.gridScanMax.y, FParams.gridScanMax.z );
		printf ( "gridDelta: %f, %f, %f\n", FParams.gridDelta.x, FParams.gridDelta.y, FParams.gridDelta.z );
		printf ( "pos: %f,%f,%f  gc: %d,%d,%d   gs: %d\n", p.x, p.y, p.z, gc.x, gc.y, gc.z, gs);
	}*/
}

// debugAccess - very useful function to show all GPU pointers
__device__ void debugAccess ()
{
	printf ( "--- gpu bufs\n" );
	for (int i=0; i < 8; i++)
		printf ( "%d: %012llx   %012llx\n", i, FBirds.bufI(i), FBirdsTmp.bufI(i) );	
}

extern "C" __global__ void countingSortFull ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// This algorithm is O(2NK) in space, O(N/P) time, where K=sizeof(Fluid)
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel). 
	
	// **NOTE** We cannot use shared memory for temporary storage since this is a 
	// global reordering and there is no synchronization across blocks. 

	int icell = FBirdsTmp.bufI(FGCELL) [ i ];	

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx; location of the particle after sort		
		int indx =  FBirdsTmp.bufI(FGNDX)  [ i ];		
	  int sort_ndx = FGrid.bufI(AGRIDOFF) [ icell ] + indx ;	// global_ndx = grid_cell_offet + particle_offset	
		
		// Transfer data to sort location	
		
		memcpy ( ((Bird*) FBirds.data(FBIRD)) + sort_ndx, ((Bird*) FBirdsTmp.data(FBIRD)) + i, sizeof(Bird) );		
		
		Bird* b = ((Bird*) FBirds.data(FBIRD)) + sort_ndx;
		Bird* bt = ((Bird*) FBirdsTmp.data(FBIRD)) + i;

		FBirds.bufI (FGCELL) [sort_ndx] =	icell;
		FBirds.bufI (FGNDX) [sort_ndx] =	indx; 		

		FGrid.bufI (AGRID) [ sort_ndx ] =	sort_ndx;			// full sort, grid indexing becomes identity				
	}
} 

extern "C" __global__ void findNeighbors ( int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = FBirds.bufI(FGCELL)[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*FAccel.gridRes.z + 1)*FAccel.gridRes.x + 1;

	register int cell, c, j, cndx;	
	register float3 dist, diri;
	register float dsq, nearest;
	Bird *bi, *bj;	
	float pi, pj;
	const float d2 = (FAccel.sim_scale * FAccel.sim_scale);
	const float rd2 = (FAccel.psmoothradius * FAccel.psmoothradius) / d2;	
	
	float birdang;
	float fov = cos ( 120 * RADtoDEG );

	// current bird
	bi = ((Bird*) FBirds.data(FBIRD)) + i;	
	bi->near_j = -1;	
	bi->nbr_cnt = 0;
	bi->ave_pos = make_float3(0,0,0);
	bi->ave_vel = make_float3(0,0,0);
	diri = normalize ( bi->vel );

	nearest = rd2;

	// check 3x3 grid cells	
	for ( c=0; c < FAccel.gridAdjCnt; c++) {
		cell = gc + FAccel.gridAdj[c];		
		// check each entry in grid..
		for ( cndx = FGrid.bufI(AGRIDOFF)[cell]; cndx < FGrid.bufI(AGRIDOFF)[cell] + FGrid.bufI(AGRIDCNT)[cell]; cndx++ ) {
			j = FGrid.bufI(AGRID)[ cndx ];				
			if (i==j) continue;

			bj = ((Bird*) FBirds.data(FBIRD)) + j;

			// check for neighbor
			dist = ( bi->pos - bj->pos );		
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

			if ( dsq < rd2 ) {
				// neighbor is within check radius..

				// confirm bird is within forward field-of-view
				dist = normalize( bj->pos - bi->pos );
				birdang = dot ( diri, dist );
				
				if (birdang > FParams.fovcos ) {

					// find nearest
					dsq = sqrt(dsq);			
					if ( dsq < nearest ) {
						nearest = dsq;
						bi->near_j = j;
					}
					// average neighbors
					bi->ave_pos += bj->pos;
					bi->ave_vel += bj->vel;
					bi->nbr_cnt++;
				}
			}	
		}		
	}

	if (bi->nbr_cnt > 0 ) {
		bi->ave_pos *= (1.0f / bi->nbr_cnt );
		bi->ave_vel *= (1.0f / bi->nbr_cnt );
	}

}

__device__ uint getGridCell ( float3 pos, uint3& gc )
{	
	gc.x = (int)( (pos.x - FAccel.gridMin.x) * FAccel.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - FAccel.gridMin.y) * FAccel.gridDelta.y);
	gc.z = (int)( (pos.z - FAccel.gridMin.z) * FAccel.gridDelta.z);		
	return (int) ( (gc.y * FAccel.gridRes.z + gc.z) * FAccel.gridRes.x + gc.x);	
}

#define maxf(a,b)  (a>b ? a : b)

inline __device__ __host__ float circleDelta(float b, float a)
{
	float d = b-a;
	d = (d > 180) ? d-360 : (d<-180) ? d+360 : d;
	return d;	
}


extern "C" __global__ void advanceBirds ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	// Get current bird
	Bird* b = ((Bird*) FBirds.data(FBIRD)) + i;
	Bird* bj;
	float3 diri, dirj;
	float dist;
	float3 force, lift, drag, thrust, accel;
	float3 fwd, up, right, vaxis, angs;
	quat4 ctrlq;
	float airflow, aoa, L, pitch, yaw;

	#ifdef DEBUG_BIRD
		if (b->id == DEBUG_BIRD) {		
			printf ("---- ADVANCE START (GPU), id %d, #%d\n", b->id, i );
			printf (" orient:  %f, %f, %f, %f\n", b->orient.x, b->orient.y, b->orient.z, b->orient.w );		
			printf (" target:  %f, %f, %f\n", b->target.x, b->target.y, b->target.z );		
		}
	#endif

	//-------------- BEHAVIORS
	//
	
	b->clr = make_float4(0,0,0,0);
	
	ctrlq = quat_inverse ( b->orient );

  float3 centroid = make_float3(0,50,0);

	// Turn isolated birds toward flock centroid
	float d = b->nbr_cnt / FParams.border_cnt;
	if ( d < 1.0 ) { 
		b->clr = make_float4(0, 1, 0, 1);	
		dirj = quat_mult ( normalize ( centroid - b->pos ), ctrlq );
		yaw = atan2( dirj.z, dirj.x )*RADtoDEG;
		pitch = asin( dirj.y )*RADtoDEG;
		b->target.z +=   yaw * FParams.border_amt;
		b->target.y += pitch * FParams.border_amt;		
	}


	if ( b->nbr_cnt > 0 ) {

		// Rule 1. Avoidance
		//
		// 1a. Side neighbor avoidance
		if ( b->near_j != -1 ) {
			// get nearest bird
			bj = ((Bird*) FBirds.data(FBIRD)) + b->near_j;
			dirj = bj->pos - b->pos;
			dist = length( dirj );

			if ( dist < FParams.safe_radius ) {

				// Angular avoidance
				dirj = quat_mult( (dirj/dist), ctrlq );
				yaw = atan2( dirj.z, dirj.x) * RADtoDEG;
				pitch = asin( dirj.y ) * RADtoDEG;
				dist = fmax( 1.0f, fmin( dist*dist, 100.0f ));				
				b->target.z -= yaw *	 FParams.avoid_angular_amt / dist;
				b->target.y -= pitch * FParams.avoid_angular_amt / dist;

				// Power adjust
				L = (length(b->vel) - length(bj->vel)) * FParams.avoid_power_amt;
				b->power = FParams.avoid_power_ctr - L; // * L;

			}
		}
		if (b->power < FParams.min_power) b->power = FParams.min_power;
		if (b->power > FParams.max_power) b->power = FParams.max_power;	


		// Rule 2. Alignment
		//
		dirj = quat_mult ( normalize( b->ave_vel ), ctrlq );
		yaw = atan2( dirj.z, dirj.x) * RADtoDEG;
		pitch = asin( dirj.y ) * RADtoDEG;
		b->target.z += yaw   * FParams.align_amt;
		b->target.y += pitch * FParams.align_amt;

		// Rule 3. Cohesion
		dirj = quat_mult ( normalize( b->ave_pos - b->pos ), ctrlq );
		yaw = atan2( dirj.z, dirj.x) * RADtoDEG;
		pitch = asin( dirj.y ) * RADtoDEG;
		b->target.z += yaw   * FParams.cohesion_amt;
		b->target.y += pitch * FParams.cohesion_amt; 		
		 
	}

	//-------------- FLIGHT MODEL

	// Body orientation
	fwd   = quat_mult ( make_float3(1,0,0), b->orient );
	up    = quat_mult ( make_float3(0,1,0), b->orient );
	right = quat_mult ( make_float3(0,0,1), b->orient );

	// Direction of motion
	b->speed = length( b->vel );
	vaxis = b->vel / b->speed;
	if ( b->speed < FParams.min_speed) b->speed = FParams.min_speed;
	if ( b->speed > FParams.max_speed) b->speed = FParams.max_speed;
	if ( b->speed == 0) vaxis = fwd;

	angs = quat_to_euler ( b->orient );

	// Target corrections
	angs.z = fmodulus ( angs.z, 180.0 );
	b->target.z = fmodulus ( b->target.z, 180 );
	b->target.x = circleDelta ( b->target.z, angs.z ) * 0.5f;
	b->target.y *= FParams.pitch_decay; 
	if ( b->target.y < FParams.pitch_min ) b->target.y = FParams.pitch_min;
	if ( b->target.y > FParams.pitch_max ) b->target.y = FParams.pitch_max;	
	if ( fabs(b->target.y) < 0.0001) b->target.y = 0;

	// Roll - Control input
	// - orient the body by roll
	ctrlq = quat_from_angleaxis ( (b->target.x - angs.x) * FParams.reaction_delay, fwd );
	b->orient = quat_normalize ( quat_mult ( b->orient, ctrlq ) );

	// Pitch & Yaw - Control inputs
	// - apply 'torque' by rotating the velocity vector based on pitch & yaw inputs				
	ctrlq = quat_from_angleaxis ( circleDelta(b->target.z, angs.z) * FParams.reaction_delay, up * -1.f );
	vaxis = normalize ( quat_mult ( vaxis, ctrlq ) ); 
	ctrlq = quat_from_angleaxis ( (b->target.y - angs.y) * FParams.reaction_delay, right );
	vaxis = normalize ( quat_mult ( vaxis, ctrlq ) );

	// Adjust velocity vector
	b->vel = vaxis * b->speed;
	force = make_float3(0,0,0);	

	// Dynamic pressure		
	airflow = b->speed + dot ( FParams.wind, fwd*-1.0f );		// airflow = air over wing due to speed + external wind	
	float dynamic_pressure = 0.5f * FParams.air_density * airflow * airflow;

	// Lift force
	aoa = acos( dot(fwd, vaxis) )*RADtoDEG + 1;		// angle-of-attack = angle between velocity and body forward		
 	if (isnan(aoa)) aoa = 1;
	// CL = sin(aoa*0.2) = coeff of lift, approximate CL curve with sin
	L = sin( aoa * 0.2) * dynamic_pressure * FParams.lift_factor * 0.5;		// lift equation. L = CL (1/2 p v^2) A
	lift = up * L;
	force += lift;	

	// Drag force	
	drag = vaxis * dynamic_pressure * FParams.drag_factor  * -1.0f;			// drag equation. D = Cd (1/2 p v^2) A
	force += drag; 

	// Thrust force
	thrust = fwd * b->power;
	force += thrust;
	
	// Integrate position		
	accel = force / FParams.mass;						// body forces	
	accel += FParams.gravity;								// gravity
	accel += FParams.wind * FParams.air_density * FParams.front_area;				// wind force. Fw = w^2 p * A, where w=wind speed, p=air density, A=frontal area
	
	b->pos += b->vel * dt;

	// Boundaries
	if ( b->pos.x < FAccel.bound_min.x ) b->pos.x = FAccel.bound_max.x;
	if ( b->pos.x > FAccel.bound_max.x ) b->pos.x = FAccel.bound_min.x;
	if ( b->pos.z < FAccel.bound_min.z ) b->pos.z = FAccel.bound_max.z;
	if ( b->pos.z > FAccel.bound_max.z ) b->pos.z = FAccel.bound_min.z;	

	// Ground avoidance
	L = b->pos.y - FAccel.bound_min.y;
	if ( L < FParams.bound_soften ) {			
		L = (FParams.bound_soften - L) / FParams.bound_soften;
		b->target.y += L * FParams.avoid_ground_amt;
		b->power = FParams.avoid_ground_power;
	} 

	// Ceiling avoidance
	L = FAccel.bound_max.y - b->pos.y;
	if ( L < FParams.bound_soften ) {	
		L = (FParams.bound_soften - L) / FParams.bound_soften;
		b->target.y -= L * FParams.avoid_ceil_amt; 						
	} 

	// Integrate velocity
	b->vel += accel * dt;
	
	vaxis = normalize( b->vel );

	// Update Orientation
	// Directional stability: airplane will typically reorient toward the velocity vector
	//  see: https://en.wikipedia.org/wiki/Directional_stability
	// this is an assumption yet much simpler/faster than integrating body orientation
	// this way we dont need torque, angular vel, or rotational inertia.
	// stalls are possible but not flat spins or 3D flying		
	ctrlq = quat_rotation_fromto ( fwd, vaxis, FParams.dynamic_stability  );
	if ( !isnan(ctrlq.x) ) {
		b->orient = quat_normalize( quat_mult ( b->orient, ctrlq ) );
	}

	#ifdef DEBUG_BIRD
		if (b->id == DEBUG_BIRD) {
			printf ("---- ADVANCE END (GPU), id %d, #%d\n", b->id, i );
			printf (" speed:   %f\n", b->speed );
			printf (" airflow: %f\n", airflow );
			printf (" orients: %f, %f, %f, %f\n", b->orient.x, b->orient.y, b->orient.z, b->orient.w );		
			printf (" angs:    %f, %f, %f\n", angs.x, angs.y, angs.z );		
			printf (" target:  %f, %f, %f\n", b->target.x, b->target.y, b->target.z );		
		}
	#endif
}


extern "C" __global__ void prefixFixup(uint *input, uint *aux, int len)
{
	unsigned int t = threadIdx.x;
	unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	if (start < len)					input[start] += aux[blockIdx.x];
	if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

extern "C" __global__ void prefixSum(uint* input, uint* output, uint* aux, int len, int zeroff)
{
	__shared__ uint scan_array[SCAN_BLOCKSIZE << 1];
	unsigned int t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	unsigned int t2 = t1 + SCAN_BLOCKSIZE;

	// Pre-load into shared memory
	scan_array[threadIdx.x] = (t1<len) ? input[t1] : 0.0f;
	scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0.0f;
	__syncthreads();

	// Reduction
	int stride;
	for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * SCAN_BLOCKSIZE)
			scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction
	for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * SCAN_BLOCKSIZE)
			scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}
	__syncthreads();

	// Output values & aux
	if (t1 + zeroff < len)	output[t1 + zeroff] = scan_array[threadIdx.x];
	if (t2 + zeroff < len)	output[t2 + zeroff] = (threadIdx.x == SCAN_BLOCKSIZE - 1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];
	if (threadIdx.x == 0) {
		if (zeroff) output[0] = 0;
		if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];
	}
}

