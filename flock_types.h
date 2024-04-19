
#ifndef DEF_FLOCK_TYPES
	#define DEF_FLOCK_TYPES

	// Particle data
	#define FBIRD		0		
	#define FGCELL		1
	#define FGNDX		2	
	#define FPREDATOR		3	// ***
	#define FGCELL_pred     4	// ***
	#define FGNDX_pred      6	// ***

	#define MAX_FLOCKS		50

	// Acceleration grid data
	#define AGRID			0	
	#define AGRIDCNT		1
	#define	AGRIDOFF		2
	#define AAUXARRAY1		3
	#define AAUXSCAN1		4
	#define AAUXARRAY2		5
	#define AAUXSCAN2		6
	#define AGRID_pred      9
	#define AGRIDCNT_pred	10

	#define GRID_UNDEF				2147483647			// max int
	#define SCAN_BLOCKSIZE		512		

	// GPU Kernels
	#define KERNEL_INSERT						0
	#define	KERNEL_COUNTING_SORT		1
	#define KERNEL_FIND_NBRS				2
	#define KERNEL_ADVANCE_ORIENT		3
	#define KERNEL_ADVANCE_VECTORS	4
	#define KERNEL_FPREFIXSUM				5
	#define KERNEL_FPREFIXFIXUP			6
	#define KERNEL_MAX							7	

	#ifdef CUDA_KERNEL
		#include "quaternion.cuh"
		typedef float3			f3;
		typedef float4			f4;
		typedef int3			i3;	
		typedef quat4			q4;		
		#define ALIGN(n)		__align__(n)									// NVCC
	#else
		#include "quaternion.h"
		#include "vec.h"		
		typedef Vec4F			f4;
		typedef Vec3F			f3;		
		typedef Vec3I			i3;
		typedef Quaternion		q4;
		#define ALIGN(n)		__declspec(align(n))					// MSVC
		//#define ALIGN(n)  __attribute__((aligned(n)))		// GCC
	#endif


	// *NOTE*
	// Bird structure used for both CPU and GPU.
  // For GPU, a struct must follow memory alignment rules.
  // This includes each 4-component member var (float4, quat4) must have 16 bytes alignment.
  // So we arrange those in the struct first, and ensure other align to 16 bytes.
	//
	struct ALIGN(16) Bird {
		
		q4			orient;
		f4			clr;

		f3			pos, vel, accel, target;
		f3			ave_pos, ave_vel;
		f3			ang_accel, ang_offaxis;
		f3			lift, drag, thrust, gravity;
		
		int			id, near_j, t_nbrs, r_nbrs;
		float		speed, pitch_adv, power;	
		float		Plift, Pdrag, Pfwd, Pturn, Ptotal;		
	};

	// entire flock states

	struct ALIGN(16) Flock {
		
		f3			centroid;
		float		speed;
		float		Plift, Pdrag, Pfwd, Pturn, Ptotal;

		int			num_flocks;
		f3			flock_centers[ MAX_FLOCKS ];
	};

	enum predState {
		HOVER,		// state1
		ATTACK,		// state2
		FOLLOW		// state3
	};

	struct ALIGN(16) Predator {	// **** struct for predator

		q4			orient;
		f4			clr;

		f3			pos, vel, accel, target;
		f3			ave_pos, ave_vel, ang_accel;

		float		speed, pitch_adv, power;
		int			id, near_j, t_nbrs, r_nbrs;

		predState			currentState;

		// //bool 		centroidReached;
		// int	 		near_bird_index;
		// float		distance_to_nearest_bird;
	};

	struct ALIGN(32) Accel {
		f3			bound_min, bound_max;
		float		psmoothradius, sim_scale;
		float		grid_size, grid_density;
		f3			gridSize, gridDelta, gridMin, gridMax;
		i3			gridRes, gridScanMax;
		int			gridSrch, gridTotal, gridAdjCnt, gridActive;
		int			gridAdj[64];	

		// gpu
		int			numThreads, numBlocks;
		int			gridThreads, gridBlocks;	
		int			szPnts;
	};

	struct ALIGN(32) Params {
	
		int			steps;
		int			num_birds;
		int			num_predators; 		// ****
		float		DT;
		float		mass;
		float		power;
		float		min_speed, max_speed;
		float		min_power, max_power;
		float		fov, fovcos;
		float		lift_factor;
		float		drag_factor;
		float		safe_radius;
		float		border_cnt;
		float		border_amt;
		float		avoid_angular_amt;
		float		avoid_power_amt;
		float		avoid_power_ctr;
		float		align_amt;
		float		cohesion_amt;
		float		pitch_decay;
		float		pitch_min, pitch_max;
		float		reaction_delay;
		float		dynamic_stability;
		float		air_density;		
		float		front_area;
		float		bound_soften;
		float		avoid_ground_amt;
		float		avoid_ground_power;
		float		avoid_ceil_amt;
		
		f3			gravity;
		f3			wind;

		float 	fov_pred, fovcos_pred;

		float 	pred_radius;					//----------
		float 	pred_flee_speed;			//----------
		float 	avoid_pred_angular_amt;
		float 	avoid_pred_power_amt;
		float 	avoid_pred_power_ctr;
		float		max_predspeed, min_predspeed;
		float		pred_mass;

		float		reynolds_avoidance;
		float		reynolds_cohesion;
		float		reynolds_alignment;
	};

#endif
