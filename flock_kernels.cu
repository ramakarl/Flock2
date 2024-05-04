//-----------------------------------------------------------------------------
// Flock v2
// Copyright (C) 2023. Rama Hoetzlein
//-----------------------------------------------------------------------------

#define CUDA_KERNEL

// #define DEBUG_BIRD			7					// enable for GPU printfs

#include "flock_kernels.cuh"

#include "quaternion.cuh"

#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

#include "datax.h"

__constant__ Params		FParams;
__constant__ Flock		FFlock;

__constant__ cuDataX	FBirds;				// birds
__constant__ cuDataX	FBirdsTmp;
__constant__ Accel		FAccel;
__constant__ cuDataX	FGrid;

__constant__ cuDataX	FPredators;		// predators

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


	

extern "C" __global__ void findNeighborsTopological ( int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = FBirds.bufUI(FGCELL)[ i ];
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
	int k, m;

	// topological distance
	float sort_d_nbr[12];
	int sort_j_nbr[12];
	int sort_num = 0;
	sort_d_nbr[0] = 10^5;
	sort_j_nbr[0] = -1;

	// current bird
	bi = ((Bird*) FBirds.data(FBIRD)) + i;	
	bi->near_j = -1;	
	bi->t_nbrs = 0;
	bi->r_nbrs = 0;
	bi->ave_pos = make_float3(0,0,0);
	bi->ave_vel = make_float3(0,0,0);
	bi->ave_del = make_float3(0,0,0);
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
			dist = ( bj->pos - bi->pos );		
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

			if ( dsq < rd2 ) {
				// neighbor is within check radius..

				// confirm bird is within forward field-of-view
				dsq = sqrt(dsq);
				dist /= dsq;
				birdang = dot ( diri, dist );				
				if (birdang > FParams.fovcos ) {					
					
					// put into topological sorted list					
					for (k = 0; dsq > sort_d_nbr[k] && k < sort_num;)
						k++;
					
					// only insert if bird is closer than the top N
					if (k <= sort_num) {
						// shift others down (insertion sort)
						if ( k != sort_num ) {
							for (m = sort_num-1; m >= k; m--) {
								sort_d_nbr[m+1] = sort_d_nbr[m];
								sort_j_nbr[m+1] = sort_j_nbr[m];
							}
						}
						
						sort_d_nbr[k] = dsq;
						sort_j_nbr[k] = j;
						
						// max topological neighbors
						if (++sort_num > 7 ) sort_num = 7;
					}	

					// count boundary neighbors
					bi->r_nbrs++;
					
				}
			}	
		}		
	}

	// compute nearest and average among N (~7) topological neighbors
	for (k=0; k < sort_num; k++) {
		bj = ((Bird*) FBirds.data(FBIRD)) + sort_j_nbr[k];
		bi->ave_pos += bj->pos;
		bi->ave_vel += bj->vel;		
		bi->ave_del += normalize (bj->pos - bi->pos) ;	
	}
	bi->near_j = sort_j_nbr[0];

	bi->t_nbrs = sort_num;
	if (sort_num > 0 ) {
		bi->ave_pos *= (1.0f / sort_num );
		bi->ave_vel *= (1.0f / sort_num );
		bi->ave_del *= (1.0f / sort_num );
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
	float del_sum = 0;

	// current bird
	bi = ((Bird*) FBirds.data(FBIRD)) + i;	
	bi->near_j = -1;	
	bi->r_nbrs = 0;
	bi->t_nbrs = 0;
	bi->ave_pos = make_float3(0,0,0);
	bi->ave_vel = make_float3(0,0,0);
	bi->ave_del = make_float3(0,0,0);
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
			dist = ( bj->pos - bi->pos );		
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

			if ( dsq < rd2 ) {
				// neighbor is within check radius..

				// confirm bird is within forward field-of-view
				dsq = sqrt( dsq );
				dist /= dsq;
				birdang = dot ( diri, dist );
				
				if (birdang > FParams.fovcos ) {

					// find nearest
					
					if ( dsq < nearest ) {
						nearest = dsq;
						bi->near_j = j;
					}
					// average neighbors
					bi->ave_del += dist;
					bi->ave_pos += bj->pos;
					bi->ave_vel += bj->vel;
					bi->r_nbrs++;
				}
			}	
		}		
	}

	if (bi->r_nbrs > 0 ) {
		bi->ave_del *= (1.0f / bi->r_nbrs);
		bi->ave_pos *= (1.0f / bi->r_nbrs);
		bi->ave_vel *= (1.0f / bi->r_nbrs);
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


extern "C" __global__ void advanceByOrientation ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	uint gc = FBirds.bufUI(FGCELL)[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range

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

  float3 center = make_float3(0,50,0);

	if ( b->r_nbrs > 0 ) {

		// Rule 1. Avoidance		
		// (Hoetzlein, orientation-based avoidance, derived from Reynolds pos-based)
		//
		// 1a. Side neighbor avoidance
		if ( b->near_j != -1 ) {
			// get nearest bird
			bj = ((Bird*) FBirds.data(FBIRD)) + b->near_j;
			dirj = bj->pos - b->pos;
			dist = length( dirj );

			//if ( dist < FParams.safe_radius ) {

				// Angular avoidance
				dirj = quat_mult( (dirj/dist), ctrlq );
				yaw = atan2( dirj.z, dirj.x) * RADtoDEG;
				pitch = asin( dirj.y ) * RADtoDEG;
				dist = fmax( 1.0f, dist * dist );
				b->target.z -= yaw *	 FParams.avoid_angular_amt / dist;
				b->target.y -= pitch * FParams.avoid_angular_amt / dist;									
			//}			

			// Power adjust
			L = length(b->vel - bj->vel) * FParams.avoid_power_amt;			
			b->power = FParams.power - L * L;
		}

		if (b->power < FParams.min_power) b->power = FParams.min_power;
		if (b->power > FParams.max_power) b->power = FParams.max_power;	


		// Rule 2. Alignment
		// (Hoetzlein, orientation-based alignment, derived from Reynolds pos-based)
		//
		dirj = quat_mult ( normalize( b->ave_vel ), ctrlq );
		yaw = atan2( dirj.z, dirj.x) * RADtoDEG;
		pitch = asin( dirj.y ) * RADtoDEG;
		b->target.z += yaw   * FParams.align_amt;
		b->target.y += pitch * FParams.align_amt;

		// Rule 3. Cohesion
		// (Hoetzlein, orientation-based cohesion, derived from Reynolds pos-based)
		//
		dirj = quat_mult ( normalize( b->ave_pos - b->pos ), ctrlq );
		yaw = atan2( dirj.z, dirj.x) * RADtoDEG;
		pitch = asin( dirj.y ) * RADtoDEG;
		b->target.z += yaw   * FParams.cohesion_amt;
		b->target.y += pitch * FParams.cohesion_amt; 		

		// Rule 4. Boundary Term
		// (Hoetzlein, new boundary term for periphery avoidance, 2023)	
		if ( FParams.boundary_cnt > 0 && b->r_nbrs <  FParams.boundary_cnt ) { 
			b->clr = make_float4(1, .5, 0, 1);	
			//dirj = quat_mult ( normalize ( FFlock.centroid - b->pos ), ctrlq );
			dirj = quat_mult ( normalize ( center - b->pos ), ctrlq );
			yaw = atan2( dirj.z, dirj.x )*RADtoDEG;
			pitch = asin( dirj.y )*RADtoDEG;
			float d = (FParams.boundary_cnt - b->r_nbrs ) / FParams.boundary_cnt;
			b->target.z +=   yaw * FParams.boundary_amt * d;
			b->target.y += pitch * FParams.boundary_amt * d;		
		}

		// Rule 5. Bird-Predators avoidance
		// (from Noortje Hagelaars, based on CPU version, 2024)
		/* Predator* p;
		for (int m = 0; m < FParams.num_predators; m++) {

			p = (Predator*) FPredators.data(FPREDATOR) + m;			
			float3 predatorDir = p->pos - b->pos;
			float predatorDist = length ( predatorDir );

			if (predatorDist < FParams.pred_radius) {
				// Flee from predator							
				predatorDir = quat_mult ( normalize ( predatorDir ), ctrlq );
				yaw = atan2(predatorDir.z, predatorDir.x) * RADtoDEG;
				pitch = asin(predatorDir.y) * RADtoDEG;
				predatorDist = fmax(1.0f, fmin(predatorDist * predatorDist, 100.0f));
				b->target.z -= yaw * FParams.avoid_pred_angular_amt; // / predatorDist;
				b->target.y -= pitch * FParams.avoid_pred_angular_amt; // / predatorDist;
				b->clr = make_float4(1, 0, 1, 1);				
			}
		}	 */ 
	}	

	//-------------- FLIGHT MODEL

	// Body orientation
	fwd   = quat_mult ( make_float3(1,0,0), b->orient );
	up    = quat_mult ( make_float3(0,1,0), b->orient );
	right = quat_mult ( make_float3(0,0,1), b->orient );

	force = make_float3(0,0,0);	
	b->thrust = make_float3(0,0,0);

	// Direction of motion
	b->speed = length( b->vel );
	vaxis = b->vel / b->speed;
	b->power = 1.0;
	if ( b->speed < FParams.min_speed) {
		//b->speed = FParams.min_speed;
		//b->thrust += vaxis * (FParams.min_speed - b->speed) * FParams.mass / dt;
		L = FParams.min_speed / b->speed;
		b->power = L; // * L;
	} else if ( b->speed > FParams.max_speed) {
		//b->speed = FParams.max_speed;
		//b->thrust += vaxis * (FParams.max_speed - b->speed) * FParams.mass / dt;
		L = FParams.max_speed / b->speed;
		b->power = L; // * L;
	}
	if ( b->speed == 0) vaxis = fwd;

	angs = quat_to_euler ( b->orient );

	// Target corrections
	angs.z = fmodulus ( angs.z, 180.0 );
	b->target.z = fmodulus ( b->target.z, 180 );
	b->target.x = circleDelta ( b->target.z, angs.z ) * 0.5f;
	b->target.y *= FParams.pitch_decay; 
	if ( b->target.y < FParams.pitch_min ) b->target.y = FParams.pitch_min;
	if ( b->target.y > FParams.pitch_max ) b->target.y = FParams.pitch_max;	
	
	// if ( fabs(b->target.y) < 0.0001) b->target.y = 0;

	// Compute angular acceleration
	// - as difference between current direction and desired direction
	b->ang_accel.x = (b->target.x - angs.x);
	b->ang_accel.y = (b->target.y - angs.y);
	b->ang_accel.z = circleDelta(b->target.z, angs.z);

	// Roll - Control input
	// - orient the body by roll
	float rx = FParams.DT*1000.0f / FParams.reaction_speed;
	ctrlq = quat_from_angleaxis ( b->ang_accel.x * rx, fwd );
	b->orient = quat_normalize ( quat_mult ( b->orient, ctrlq ) );

	// Pitch & Yaw - Control inputs
	// - apply 'torque' by rotating the velocity vector based on pitch & yaw inputs						
	ctrlq = quat_from_angleaxis ( b->ang_accel.z * rx, up * -1.f );
	vaxis = normalize ( quat_mult ( vaxis, ctrlq ) ); 
	ctrlq = quat_from_angleaxis ( b->ang_accel.y * rx, right );
	vaxis = normalize ( quat_mult ( vaxis, ctrlq ) );

	// Adjust velocity vector
	b->vel = vaxis * b->speed;

	// Compute off-axis angle from neighborhood average	
	f3 v0, v1;
	v0 = normalize ( b->vel );
	v1 = normalize ( b->ave_vel );
	ctrlq = quat_rotation_fromto ( v0, v1, 1.0 );	
	b->ang_offaxis = quat_to_euler ( ctrlq ) ;

	// Dynamic pressure		
	airflow = b->speed + dot ( FParams.wind, fwd*-1.0f );		// airflow = air over wing due to speed + external wind	
	float dynamic_pressure = 0.5f * FParams.air_density * airflow * airflow;

	// Lift force
	//-- dynamic CL
	// aoa = acos( dot(fwd, vaxis) )*RADtoDEG + 1;		// angle-of-attack = angle between velocity and body forward		
 	// if (isnan(aoa)) aoa = 1;	
	// L = (sin( aoa * 0.1)+0.5) * dynamic_pressure * FParams.lift_factor * FParams.wing_area;		// lift equation. L = CL (1/2 p v^2) A
	//-- fixed CL
	L = dynamic_pressure * FParams.lift_factor * FParams.wing_area;		// lift equation. L = CL (1/2 p v^2) A

	b->lift = up * L;
	force += b->lift;	

	// Drag force	
	b->drag = vaxis * dynamic_pressure * -FParams.drag_factor  * FParams.wing_area;			// drag equation. D = Cd (1/2 p v^2) A
	force += b->drag; 

	// Thrust force
	b->thrust += fwd * b->power * FParams.power;
	force += b->thrust;
	
	// Gravity force
	b->gravity = FParams.gravity * FParams.mass;		// Fgrav = mg
	force += b->gravity;	

	// Ground avoidance
	L = b->pos.y - FAccel.bound_min.y;
	if ( L < FParams.bound_soften ) {			
		L = (FParams.bound_soften - L) / FParams.bound_soften;
		//force.y += L * FParams.avoid_ground_amt;
		b->target.y += L * FParams.avoid_ground_amt;		
	} 
	// Ceiling avoidance
	L = FAccel.bound_max.y - b->pos.y;
	if ( L < FParams.bound_soften ) {	
		L = (FParams.bound_soften - L) / FParams.bound_soften;
		//force.y -= L * FParams.avoid_ground_amt;
		b->target.y -= L * FParams.avoid_ceil_amt; 						
	} 

	// Compute energy used (stats, read only)
	// w = F . d											// Work is force applied over displacement
	// P = w/t												// Power is work divided by time
	// P = W V												// from Pennycuick, where W = weight in Newtons = mg, V = velocity
	b->Plift = length( b->lift ) * b->speed;			// lift is force applied to move air downward
	b->Pdrag = length( b->drag ) * b->speed;			// drag is force against motion (profile + parasitic drag)
	// compute force vector, after eliminating lift, drag, gravity
	float Fresidual = length(force - b->lift - b->drag - b->gravity);
	f3 delta_v = (force - b->lift - b->drag - b->gravity) * dt / FParams.mass;	
	float vdotv = dot ( delta_v, vaxis );
	b->Pfwd = Fresidual * vdotv;															// energy for forward acceleration (beyond drag)
	b->Pturn = Fresidual * length( delta_v - vdotv * vaxis );	 // energy for turning 
	// total energy/bird = lift + drag + fwd energy + turn energy
	b->Ptotal = b->Plift + b->Pdrag + b->Pfwd + b->Pturn;

	//float cl = min( (b->Pfwd-0.0187) * 10000.0f, 1.0);
	//float cl = min( b->Pturn * 1000.0f, 1.0);
	//b->clr = make_float4( 1-cl, cl, 0, 1);

	// Integrate position		
	accel = force / FParams.mass;						// body forces	
	accel += FParams.wind * FParams.air_density * FParams.front_area;				// wind force. Fw = w^2 p * A, where w=wind speed, p=air density, A=frontal area
	
	b->pos += b->vel * dt;

	// Boundaries
	if ( b->pos.x < FAccel.bound_min.x ) b->pos.x = FAccel.bound_max.x;
	if ( b->pos.x > FAccel.bound_max.x ) b->pos.x = FAccel.bound_min.x;
	if ( b->pos.z < FAccel.bound_min.z ) b->pos.z = FAccel.bound_max.z;
	if ( b->pos.z > FAccel.bound_max.z ) b->pos.z = FAccel.bound_min.z;	

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

extern "C" __global__ void advanceByVectors ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	// Reynold's classic vector-based Boids
	//

	// Get current bird
	Bird* b = ((Bird*) FBirds.data(FBIRD)) + i;
	Bird* bj;
	float3 dirj, force, accel, angs, v0, v1;
	quat4 q;

	force = make_float3(0,0,0);
	b->clr = make_float4(0,0,0,0);			// default, visualize ang accel (w=0)

	b->speed = length( b->vel );
	
	if ( b->r_nbrs > 0 ) {
		
		bj = ((Bird*) FBirds.data(FBIRD)) + b->near_j;

		// Rule #1. Avoidance	(Reynolds position-based)		
		dirj = b->ave_del;
		force -= dirj * FParams.reynolds_avoidance;
		/*if ( b->near_j != -1 ) {				
			dirj = bj->pos - b->pos;
			force -= dirj * FParams.reynolds_avoidance;
		}*/
	
		// Rule #2. Alignment	(Reynolds position-based)			
		dirj = b->ave_vel - b->vel;
		force += dirj * FParams.reynolds_alignment;

		// Rule #3. Cohesion (Reynolds position-based)			
		dirj = b->ave_pos - b->pos;
		force += dirj * FParams.reynolds_cohesion;
	}
	
	// Gravity force
	b->gravity = FParams.gravity * FParams.mass;		// Fgrav = mg
	//force += b->gravity;
	// Lift force - exactly equal to gravity in Reynold's model
	//force -= b->gravity;
	
	// Integrate position	& velocity
	accel = force / FParams.mass;						// body forces	
	v0 = normalize ( b->vel );	

	// [stats only] Compute energy used
	// w = F . d											// Work is force applied over displacement
	// P = w/t												// Power is work divided by time
	// P = W V												// from Pennycuick, where W = weight in Newtons = mg, V = velocity
	// -- compute lift, does not affect Reynolds sim
	float airflow = b->speed;			// airflow = air over wing due to speed + external wind	
	float dynamic_pressure = 0.5f * FParams.air_density * airflow * airflow;	
	// assume constant CL = 1.25
	f3 vaxis = b->vel / b->speed;			// normalized direction of velocity 
	float L = dynamic_pressure * FParams.lift_factor * FParams.wing_area;			// lift equation. L = CL (1/2 p v^2) A	
	b->lift = make_float3(0,1,0) * L;
	// -- compute drag, does not affect Reynolds sim
	float D = dynamic_pressure * FParams.drag_factor  * FParams.wing_area;		// drag equation. D = Cd (1/2 p v^2) A
	b->drag = vaxis * -D;
	// -- compute gravity, does not affect Reynolds sim	
	b->gravity = FParams.gravity * FParams.mass;		// Fgrav = mg
	// -- compute energies
	b->Plift = L * b->speed;					// lift is force applied to move air downward
	b->Pdrag = D * b->speed;					// drag is force against motion (profile + parasitic drag)
	// compute force vector, after eliminating lift, drag, gravity
	float Fresidual = length(force - b->lift - b->drag - b->gravity);
	f3 delta_v = (force - b->lift - b->drag - b->gravity) * dt / FParams.mass;		
	float vdotv = dot ( delta_v, vaxis );
	b->Pfwd = Fresidual * vdotv;															// energy for forward acceleration (beyond drag)
	b->Pturn = Fresidual * length( delta_v - vdotv * vaxis );	 // energy for turning 
	// total energy/bird = lift + drag + fwd energy + turn energy
	b->Ptotal = b->Plift + b->Pdrag + b->Pfwd + b->Pturn;

	// [stats only] visualize turn energy	
	// float cl = min( b->Pturn * 100.0f, 1.0);
	// b->clr = make_float4( 1-cl, cl, 0, 1);	

	// Integrate position and velocity 
	b->vel += accel * dt;
	b->pos += b->vel * dt;

	// Speed limit
	b->speed = length( b->vel );	
	if ( b->speed < FParams.min_speed) b->speed = FParams.min_speed;
	if ( b->speed > FParams.max_speed) b->speed = FParams.max_speed;
	b->vel = normalize(b->vel) * b->speed;

	//b->vel.y *= 0.9999;
	
	// Orient the bird (for rendering)
	b->orient = quat_from_directionup ( v0, make_float3(0,1,0) );

	// Wrap boundaries (X/Z)
	if ( b->pos.x < FAccel.bound_min.x ) b->pos.x = FAccel.bound_max.x;
	if ( b->pos.x > FAccel.bound_max.x ) b->pos.x = FAccel.bound_min.x;
	if ( b->pos.z < FAccel.bound_min.z ) b->pos.z = FAccel.bound_max.z;
	if ( b->pos.z > FAccel.bound_max.z ) b->pos.z = FAccel.bound_min.z;	

	if ( b->pos.y < FAccel.bound_min.y ) b->pos.y = FAccel.bound_max.y;
	if ( b->pos.y > FAccel.bound_max.y ) b->pos.y = FAccel.bound_min.y;	

	// [stats only] Compute angular acceleration, does not affect sim
	
	v1 = normalize ( b->vel );
	q = quat_rotation_fromto ( v0, v1, 1.0 );	
	b->ang_accel = quat_to_euler ( q );

	// [stats only] Compute angle from neighborhood average	
	v1 = normalize ( b->ave_vel );
	q = quat_rotation_fromto ( v0, v1, 1.0 );	
	b->ang_offaxis = quat_to_euler ( q ) ;

	#ifdef DEBUG_BIRD
		if (b->id == DEBUG_BIRD) {
			float aa = length ( b->ang_accel );
			printf ("---- ADVANCE END (GPU), id %d, #%d\n", b->id, i );			
			printf (" force:     %f, %f, %f\n", force.x, force.y, force.z);
			printf (" ang_accel: %f\n", aa );
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

