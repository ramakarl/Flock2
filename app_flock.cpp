//--------------------------------------------------------
//
// Flock v2
// Rama Hoetzlein, 2023
//
//--------------------------------------------------------------------------------
// Copyright 2023 (c) Quanta Sciences, Rama Hoetzlein, ramakarl.com
//
// * Derivative works may append the above copyright notice but should not remove or modify earlier notices.
//
// MIT License:

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
// associated documentation files (the "Software"), to deal in the Software without restriction, including without 
// limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
// and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF 
// OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <time.h>
#include "main.h"					// window system 
#include "timex.h"				// for accurate timing
#include "quaternion.h"
#include "datax.h"
#include "mersenne.h"
#include "common_cuda.h"
#include "geom_helper.h"

#define DEBUG_CUDA		true	
//#define DEBUG_BIRD		7

#include "gxlib.h"			// low-level render
#include "g2lib.h"			// gui system
using namespace glib;

// Bird structures
//
#include "flock_types.h"

struct vis_t {
	vis_t( Vec3F p, float r, Vec4F c) {pos=p; radius=r; clr=c;}
	Vec3F		pos;
	float		radius;
	Vec4F		clr;
};
struct graph_t {
	int			x;
	float		y[2048];
	Vec4F		clr;
};
#define GRAPH_BANK		0
#define GRAPH_PITCH		1
#define GRAPH_VEL			2
#define GRAPH_ACCEL		3
#define GRAPH_MAX			4

// Application
//
class Sample : public Application {
public:
	virtual bool init();
	virtual void startup ();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion (AppEnum button, int x, int y, int dx, int dy);	
	virtual void keyboard(int keycode, AppEnum action, int mods, int x, int y);
	virtual void mouse (AppEnum button, AppEnum state, int mods, int x, int y);	
	virtual void mousewheel(int delta);
	virtual void shutdown();

	Bird*			AddBird ( Vec3F pos, Vec3F vel, Vec3F target, float power );	
	
	// simulation
	void			DefaultParams ();
	void			Reset (int num_bird );	
	void		  Run ();
	void			FindNeighbors ();	
	void			Advance ();		
	
	// rendering
	void			SelectBird (float x, float y);
	void			InitGraphs ();
	void			VisualizeSelectedBird ();
	void			DebugBird ( int id, std::string msg );
	void			CameraToBird ( int b );
	void			CameraToCockpit( int b );
	void			CameraToCentroid ();
	void			drawBackground();
	void			drawGrid( Vec4F clr );

	// acceleration
	void			InitializeGrid ();
	void			InsertIntoGrid ();
	void			PrefixSumGrid ();
	void			DrawAccelGrid ();	

	DataX			m_Birds;
	DataX			m_BirdsTmp;
	DataX			m_Grid;
	Accel			m_Accel;
	Params		m_Params;

	Mersenne  m_rnd;	

	Vec3F			m_centroid;

	float			m_time;
	bool			m_run;
	int				m_cam_mode;
	bool			m_cam_adjust;
	Camera3D*	m_cam;
	Vec3F			m_cam_fwd;
	int				mouse_down;
	int			  m_bird_sel, m_bird_ndx;
	bool		  m_cockpit_view;
	bool			m_draw_sphere;
	bool		  m_draw_grid;
	bool			m_draw_vis;
	bool			m_gpu, m_kernels_loaded;

	std::vector< vis_t >  m_vis;
	std::vector< graph_t> m_graph;

	// CUDA / GPU
	#ifdef BUILD_CUDA
		void			LoadKernel ( int fid, std::string func );
		void			LoadAllKernels ();

		CUcontext		m_ctx;
		CUdevice		m_dev; 
		CUdeviceptr	m_cuAccel;
		CUdeviceptr	m_cuParam;
		CUmodule		m_Module;
		CUfunction	m_Kernel[ KERNEL_MAX ];
	#endif
};

Sample obj;

#ifdef BUILD_CUDA
	void Sample::LoadKernel ( int fid, std::string func )
	{
		char cfn[512];		
		strcpy ( cfn, func.c_str() );
		cuCheck ( cuModuleGetFunction ( &m_Kernel[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, DEBUG_CUDA );	
	}

	void Sample::LoadAllKernels ()
	{
		std::string ptxfile = "flock_kernels.ptx";
		std::string filepath;
		if (!getFileLocation ( ptxfile, filepath )) {
			printf ( "ERROR: Unable to find %s\n", ptxfile.c_str() );
			exit(-7);
		}
		cuCheck ( cuModuleLoad ( &m_Module, filepath.c_str() ), "LoadKernel", "cuModuleLoad", "flock_kernels.ptx", DEBUG_CUDA );

		LoadKernel ( KERNEL_INSERT,					"insertParticles" );
		LoadKernel ( KERNEL_COUNTING_SORT,	"countingSortFull" );	
		LoadKernel ( KERNEL_FIND_NBRS,			"findNeighborsTopological" );
		LoadKernel ( KERNEL_ADVANCE,				"advanceBirds" );
		LoadKernel ( KERNEL_FPREFIXSUM,			"prefixSum" );
		LoadKernel ( KERNEL_FPREFIXFIXUP,		"prefixFixup" );
	}
#endif

Bird* Sample::AddBird ( Vec3F pos, Vec3F vel, Vec3F target, float power )
{
	Vec3F dir, angs;

	int ndx = m_Birds.AddElem ( FBIRD );

	Bird b;
	b.id = ndx;
	b.pos = pos;
	b.vel = vel;		
	b.target = target;	
	b.power = power;
	b.pitch_adv = 0;
	b.accel.Set(0,0,0);

	dir = b.vel; dir.Normalize();	
	b.orient.fromDirectionAndUp ( dir, Vec3F(0,1,0) );
	b.orient.normalize();
	b.orient.toEuler ( angs );			
	
	m_Birds.SetElem (0, ndx, &b );

	return (Bird*) m_Birds.GetElem( FBIRD, ndx);
}

int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void ComputeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = std::min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}

void Sample::DefaultParams ()
{
	// Flock parameters
	//
	// SI units:
	// vel = m/s, accel = m/s^2, mass = kg, thrust(power) = N (kg m/s^2)	
	//
	m_Params.steps =						2;
	m_Params.DT =								0.005;									// timestep (sec), .005 = 200 hz
	m_Params.mass =							0.1;										// bird mass (kg)
	m_Params.min_speed =				5;											// min speed (m/s)
	m_Params.max_speed =				18;											// max speed (m/s)
	m_Params.min_power =				-20;										// min power (N)
	m_Params.max_power =				20;											// max power (N)
	m_Params.wind =							Vec3F(0,0,0);						// wind direction & strength
	m_Params.fov =							130;										// bird field-of-view (degrees)
	m_Params.fovcos = cos ( m_Params.fov * DEGtoRAD );

	m_Params.lift_factor =			0.100;									// lift factor
	m_Params.drag_factor =			0.002;									// drag factor
	m_Params.safe_radius =			2.0;										// radius of avoidance (m)
	m_Params.border_cnt =				30;											// border width (# birds)
	m_Params.border_amt =				0.08f;									// border steering amount (keep <0.1)
	
	m_Params.avoid_angular_amt= 0.02f;									// bird angular avoidance amount
	m_Params.avoid_power_amt =	0.02f;								 		// power avoidance amount (N)
	m_Params.avoid_power_ctr =	3;											// power avoidance center (N)
	
	m_Params.align_amt =				0.700f;									// bird alignment amount

	m_Params.cohesion_amt =			0.0001f;								// bird cohesion amount

	m_Params.pitch_decay =			0.999;									// pitch decay (return to level flight)
	m_Params.pitch_min =				-40;										// min pitch (degrees)
	m_Params.pitch_max =				40;											// max pitch (degrees)
	
	m_Params.reaction_delay =		0.0020f;								// reaction delay

	m_Params.dynamic_stability = 0.6f;									// dyanmic stability factor
	m_Params.air_density =			1.225;									// air density (kg/m^3)
	m_Params.gravity =					Vec3F(0, -9.8, 0);			// gravity (m/s^2)
	m_Params.front_area =				0.1f;										// section area of bird into wind
	m_Params.bound_soften	=			20;											// ground detection range
	m_Params.avoid_ground_power = 3;										// ground avoid power setting 
	m_Params.avoid_ground_amt = 0.5f;										// ground avoid strength
	m_Params.avoid_ceil_amt =   0.1f;										// ceiling avoid strength
	
}


void Sample::Reset (int num )
{
	Vec3F pos, vel;
	float h;
	int grp;
	Bird* b;

	// Global flock variables
	//
	m_Params.num_birds = num;
	
	// Initialized bird memory
	//
	int numPoints = m_Params.num_birds;
	uchar usage = (m_gpu) ? DT_CPU | DT_CUMEM : DT_CPU;

	m_Birds.DeleteAllBuffers ();
	m_Birds.AddBuffer ( FBIRD,  "bird",		sizeof(Bird),	numPoints, usage );
	m_Birds.AddBuffer ( FGCELL, "gcell",	sizeof(uint),	numPoints, usage );
	m_Birds.AddBuffer ( FGNDX,  "gndx",		sizeof(uint),	numPoints, usage );		

	// Add birds
	//
	for (int n=0; n < numPoints; n++ ) {
		
		//-- test: head-on impact of two bird flocks
		/* bool ok = false;
		while (!ok) {
			pos = m_rnd.randV3( -50, 50 );
			if (pos.Length() < 50 ) {				
				grp = (n % 2);
				pos += Vec3F( 0, 100, grp ? -60 : 60 );
				vel = Vec3F(  0,   0, grp ?  50 :-50 );
				h = grp ? 90 : -90;
				b = AddBird ( pos, vel, Vec3F(0, 0, h), 3); 
				//rb->clr = (grp==0) ? Vec4F(1,0,0,1) : Vec4F(0,1,0,1);
				ok = true;
			}
		} */
		
		// randomly distribute birds
		pos = m_rnd.randV3( -100, 100 );
		pos.y = pos.y * .5f + 50;
		vel = m_rnd.randV3( -20, 20 );
		h = m_rnd.randF(-180, 180);
		b = AddBird ( pos, vel, Vec3F(0, 0, h), 3);  
		b->clr = Vec4F( (pos.x+100)/200.0f, pos.y/200.f, (pos.z+100)/200.f, 1.f ); 	

	}
	
	// Initialize accel grid
	//
	m_Accel.bound_min = Vec3F(-100,   0, -100);
	m_Accel.bound_max = Vec3F( 100, 100,  100);
	m_Accel.psmoothradius = 6;
	m_Accel.grid_density = 1.0;
	m_Accel.sim_scale = 1.0;

	InitializeGrid ();

	#ifdef BUILD_CUDA
		// Reset GPU 
		if (m_gpu) {
		
			// Load GPU kernels [if needed]
			if (!m_kernels_loaded) {
				m_kernels_loaded = true;
				LoadAllKernels ();
				size_t len;
				cuCheck ( cuModuleGetGlobal ( &m_cuAccel,  &len, m_Module, "FAccel" ), "Initialize", "cuModuleGetGlobal", "cuAccel", true );
				cuCheck ( cuModuleGetGlobal ( &m_cuParam, &len, m_Module, "FParams" ), "Initialize", "cuModuleGetGlobal", "cuParam", true );
			}
			// Assign GPU symbols
			m_Birds.AssignToGPU ( "FBirds", m_Module );
			m_BirdsTmp.AssignToGPU ( "FBirdsTmp", m_Module );
			m_Grid.AssignToGPU ( "FGrid", m_Module );		
			cuCheck ( cuMemcpyHtoD ( m_cuAccel, &m_Accel,	sizeof(Accel) ), "Accel", "cuMemcpyHtoD", "cuAccel", DEBUG_CUDA );
			cuCheck ( cuMemcpyHtoD ( m_cuParam, &m_Params, sizeof(Params) ), "Params", "cuMemcpyHtoD", "cuParam", DEBUG_CUDA );

			// Commit birds
			m_Birds.CommitAll ();

			// Update temp list
			m_BirdsTmp.MatchAllBuffers ( &m_Birds, DT_CUMEM );

			// Compute particle thread blocks	
			int threadsPerBlock = 512;	
			ComputeNumBlocks ( numPoints, threadsPerBlock, m_Accel.numBlocks, m_Accel.numThreads);				// particles    
			m_Accel.szPnts = (m_Accel.numBlocks  * m_Accel.numThreads);     
			dbgprintf ( "  Particles: %d, threads:%d x %d=%d, size:%d\n", numPoints, m_Accel.numBlocks, m_Accel.numThreads, m_Accel.numBlocks*m_Accel.numThreads, m_Accel.szPnts);	

			// Update GPU access
			m_Birds.UpdateGPUAccess ();
			m_BirdsTmp.UpdateGPUAccess ();
			m_Grid.UpdateGPUAccess ();

		}
	#endif

	printf ( "Added %d birds.\n", m_Params.num_birds );
}


void Sample::drawGrid( Vec4F clr )
{
	Vec3F a;
	float o = 0.02;

	// center section
	o = -0.02;			// offset
	for (int n=-5000; n <= 5000; n += 50 ) {
		drawLine3D ( Vec3F(n, o,-5000), Vec3F(n, o, 5000), clr );
		drawLine3D ( Vec3F(-5000, o, n), Vec3F(5000, o, n), clr );
	}

}


// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k * gs / d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
//
void Sample::InitializeGrid ()
{
	// Grid size - cell spacing in SPH units
	m_Accel.grid_size = m_Accel.psmoothradius / m_Accel.grid_density;	
																					
	// Grid bounds - one cell beyond fluid domain
	m_Accel.gridMin = m_Accel.bound_min;		m_Accel.gridMin -= float(2.0*(m_Accel.grid_size / m_Accel.sim_scale ));
	m_Accel.gridMax = m_Accel.bound_max;		m_Accel.gridMax += float(2.0*(m_Accel.grid_size / m_Accel.sim_scale ));
	m_Accel.gridSize = m_Accel.gridMax - m_Accel.gridMin;	
	
	float grid_size = m_Accel.grid_size;
	float world_cellsize = grid_size / m_Accel.sim_scale;		// cell spacing in world units
	float sim_scale = m_Accel.sim_scale;

	// Grid res - grid volume uniformly sub-divided by grid size
	m_Accel.gridRes.x = (int) ceil ( m_Accel.gridSize.x / world_cellsize );		// Determine grid resolution
	m_Accel.gridRes.y = (int) ceil ( m_Accel.gridSize.y / world_cellsize );
	m_Accel.gridRes.z = (int) ceil ( m_Accel.gridSize.z / world_cellsize );
	m_Accel.gridSize.x = m_Accel.gridRes.x * world_cellsize;						// Adjust grid size to multiple of cell size
	m_Accel.gridSize.y = m_Accel.gridRes.y * world_cellsize;
	m_Accel.gridSize.z = m_Accel.gridRes.z * world_cellsize;	
	m_Accel.gridDelta = Vec3F(m_Accel.gridRes) / m_Accel.gridSize;		// delta = translate from world space to cell #	
	
	// Grid total - total number of grid cells
	m_Accel.gridTotal = (int) (m_Accel.gridRes.x * m_Accel.gridRes.y * m_Accel.gridRes.z);

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_Accel.gridSrch = (int) (floor(2.0f*(m_Accel.psmoothradius / sim_scale) / world_cellsize) + 1.0f);
	if ( m_Accel.gridSrch < 2 ) m_Accel.gridSrch = 2;
	m_Accel.gridAdjCnt = m_Accel.gridSrch * m_Accel.gridSrch * m_Accel.gridSrch;
	m_Accel.gridScanMax = m_Accel.gridRes - Vec3I( m_Accel.gridSrch, m_Accel.gridSrch, m_Accel.gridSrch );

	if ( m_Accel.gridSrch > 6 ) {
		dbgprintf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}

	// Auxiliary buffers - prefix sums sizes
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = m_Accel.gridTotal;
	int numElem2 = int ( numElem1 / blockSize ) + 1;
	int numElem3 = int ( numElem2 / blockSize ) + 1;

	int numPoints = m_Params.num_birds;

	int mem_usage = (m_gpu) ? DT_CPU | DT_CUMEM : DT_CPU;

	// Allocate acceleration
	m_Grid.DeleteAllBuffers ();
	m_Grid.AddBuffer ( AGRID,		  "grid",			sizeof(uint), numPoints,					mem_usage );
	m_Grid.AddBuffer ( AGRIDCNT,	"gridcnt",	sizeof(uint), m_Accel.gridTotal,	mem_usage );
	m_Grid.AddBuffer ( AGRIDOFF,	"gridoff",	sizeof(uint), m_Accel.gridTotal,	mem_usage );
	m_Grid.AddBuffer ( AAUXARRAY1, "aux1",		sizeof(uint), numElem2,						mem_usage );
	m_Grid.AddBuffer ( AAUXSCAN1,  "scan1",		sizeof(uint), numElem2,						mem_usage );
	m_Grid.AddBuffer ( AAUXARRAY2, "aux2",		sizeof(uint), numElem3,						mem_usage );
	m_Grid.AddBuffer ( AAUXSCAN2,  "scan2",		sizeof(uint), numElem3,						mem_usage );

	for (int b=0; b <= AAUXSCAN2; b++)
		m_Grid.SetBufferUsage ( b, DT_UINT );		// for debugging

	// Grid adjacency lookup - stride to access neighboring cells in all 6 directions
	int cell = 0;
	for (int y=0; y < m_Accel.gridSrch; y++ ) 
		for (int z=0; z < m_Accel.gridSrch; z++ ) 
			for (int x=0; x < m_Accel.gridSrch; x++ ) 
				m_Accel.gridAdj [ cell++]  = ( y * m_Accel.gridRes.z+ z ) * m_Accel.gridRes.x +  x ;			

	// Done
	dbgprintf ( "  Accel Grid: %d, Res: %dx%dx%d\n", m_Accel.gridTotal, (int) m_Accel.gridRes.x, (int) m_Accel.gridRes.y, (int) m_Accel.gridRes.z );		
}


void Sample::InsertIntoGrid ()
{
	int numPoints = m_Params.num_birds;

	if (m_gpu) {

		#ifdef BUILD_CUDA
			// Reset all grid cells to empty	
			cuCheck ( cuMemsetD8 ( m_Grid.gpu(AGRIDCNT),	0,	m_Accel.gridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemsetD8", "AGRIDCNT", DEBUG_CUDA );
			cuCheck ( cuMemsetD8 ( m_Grid.gpu(AGRIDOFF),	0,	m_Accel.gridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemsetD8", "AGRIDOFF", DEBUG_CUDA );
			cuCheck ( cuMemsetD8 ( m_Birds.gpu(FGCELL),		0,	numPoints*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGCELL", DEBUG_CUDA );
			cuCheck ( cuMemsetD8 ( m_Birds.gpu(FGNDX),		0,	numPoints*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGNDX", DEBUG_CUDA );

			// Insert into grid (GPU)
			void* args[1] = { &numPoints };
			cuCheck(cuLaunchKernel ( m_Kernel[KERNEL_INSERT], m_Accel.numBlocks, 1, 1, m_Accel.numThreads, 1, 1, 0, NULL, args, NULL),
				"InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", DEBUG_CUDA );
		#endif

	} else {

		// Insert into grid 
		// Reset all grid cells to empty		
		memset( m_Grid.bufUI(AGRIDCNT),	0,	m_Accel.gridTotal*sizeof(uint));
		memset( m_Grid.bufUI(AGRIDOFF),	0,	m_Accel.gridTotal*sizeof(uint));
		memset( m_Birds.bufUI(FGCELL),	0,	numPoints*sizeof(int));
		memset( m_Birds.bufUI(FGNDX),		0,	numPoints*sizeof(int));

		float poff = m_Accel.psmoothradius / m_Accel.sim_scale;

		// Insert each particle into spatial grid
		Vec3F gcf;
		Vec3I gc;
		int gs; 
		Vec3F ppos;
		uint* pgcell =	  m_Birds.bufUI (FGCELL);
		uint* pgndx =			m_Birds.bufUI (FGNDX);		

		Bird* b;
	
		for ( int n=0; n < numPoints; n++ ) {		
		
			b = (Bird*) m_Birds.GetElem( FBIRD, n);
			ppos = b->pos;

			gcf = (ppos - m_Accel.gridMin) * m_Accel.gridDelta; 
			gc = Vec3I( int(gcf.x), int(gcf.y), int(gcf.z) );
			gs = (gc.y * m_Accel.gridRes.z + gc.z)*m_Accel.gridRes.x + gc.x;
	
			if ( gc.x >= 1 && gc.x <= m_Accel.gridScanMax.x && gc.y >= 1 && gc.y <= m_Accel.gridScanMax.y && gc.z >= 1 && gc.z <= m_Accel.gridScanMax.z ) {
				*pgcell = gs;
				*pgndx = *m_Grid.bufUI(AGRIDCNT, gs);
				(*m_Grid.bufUI(AGRIDCNT, gs))++;			
			} else {
				*pgcell = GRID_UNDEF;				
			}					
			pgcell++;
			pgndx++;		
		}
	}

}

void Sample::PrefixSumGrid ()
{
	if (m_gpu) {

		#ifdef BUILD_CUDA
			// PrefixSum - GPU
			// Prefix Sum - determine grid offsets
			int blockSize = SCAN_BLOCKSIZE << 1;
			int numElem1 = m_Accel.gridTotal;		
			int numElem2 = int ( numElem1 / blockSize ) + 1;
			int numElem3 = int ( numElem2 / blockSize ) + 1;
			int threads = SCAN_BLOCKSIZE;
			int zero_offsets = 1;
			int zon = 1;

			CUdeviceptr array1  = m_Grid.gpu(AGRIDCNT);		// input
			CUdeviceptr scan1   = m_Grid.gpu(AGRIDOFF);		// output
			CUdeviceptr array2  = m_Grid.gpu(AAUXARRAY1);
			CUdeviceptr scan2   = m_Grid.gpu(AAUXSCAN1);
			CUdeviceptr array3  = m_Grid.gpu(AAUXARRAY2);
			CUdeviceptr scan3   = m_Grid.gpu(AAUXSCAN2);

			if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
				dbgprintf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
			}
			// prefix scan in blocks with up to two hierarchy layers
			// this allows total # elements up to SCAN_BLOCKSIZE^3 = 512^3 = 134 million max
			void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets }; // sum array1. output -> scan1, array2
			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM:A", DEBUG_CUDA );

			void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon }; // sum array2. output -> scan2, array3
			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM:B", DEBUG_CUDA );

			if ( numElem3 > 1 ) {
				CUdeviceptr nptr = {0};
				void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	// sum array3. output -> scan3
				cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP:C", DEBUG_CUDA );

				void* argsD[3] = { &scan2, &scan3, &numElem2 };	// merge scan3 into scan2. output -> scan2
				cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP:D", DEBUG_CUDA );
			}

			void* argsE[3] = { &scan1, &scan2, &numElem1 };		// merge scan2 into scan1. output -> scan1
			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP:E", DEBUG_CUDA );
			// returns grid offsets: scan1 => AGRIDOFF

			// Counting Sort
			//
			// transfer particle data to temp buffers 
			//  (required by gpu counting sort algorithm, gpu-to-gpu copy, no context sync needed)	
			m_Birds.CopyAllBuffers ( &m_BirdsTmp, DT_CUMEM );
		
			// sort
			int numPoints = m_Params.num_birds;
			void* args[1] = { &numPoints };
			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_COUNTING_SORT], m_Accel.numBlocks, 1, 1, m_Accel.numThreads, 1, 1, 0, NULL, args, NULL),
						"CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", DEBUG_CUDA );
		#endif

	} else {

		// PrefixSum - CPU
		// cpu scan and sort is implemented to give identical output as gpu version,
		// *except* that birds are not deep copied for cache coherence as they are on gpu.
		// the grid cells will contain the same list of points in either case.
		int numPoints = m_Params.num_birds;
		int numCells = m_Accel.gridTotal;
		uint* mgrid = (uint*) m_Grid.bufI(AGRID);
		uint* mgcnt = (uint*) m_Grid.bufI(AGRIDCNT);
		uint* mgoff = (uint*) m_Grid.bufI(AGRIDOFF);

		// compute prefix sums for offsets
		int sum = 0;	
		for (int n=0; n < numCells; n++) {
			mgoff[n] = sum;
			sum += mgcnt[n];
		}

		// compute master grid list
		uint* pgcell =	  m_Birds.bufUI (FGCELL);
		uint* pgndx =			m_Birds.bufUI (FGNDX);		
		int gs, sort_ndx;
		for (int k=0; k < numPoints; k++) {
			mgrid[k] = GRID_UNDEF;
		}
		for (int j=0; j < numPoints; j++) {

			if ( *pgcell != GRID_UNDEF ) {			
				sort_ndx = mgoff [ *pgcell ] + *pgndx;
				mgrid[ sort_ndx ] = j;			
			} 
			pgcell++;
			pgndx++;
		}
	}
}

void Sample::FindNeighbors ()
{

	if (m_gpu) {

		#ifdef BUILD_CUDA
			// Find neighborhood (GPU)
			//		
			int numPoints = m_Params.num_birds;
			void* args[1] = { &numPoints };
			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_FIND_NBRS],  m_Accel.numBlocks, 1, 1, m_Accel.numThreads, 1, 1, 0, NULL, args, NULL), "FindNeighbors", "cuLaunch", "FUNC_FIND_NBRS", DEBUG_CUDA );
		#endif

	} else {

		// Find neighborhood of each bird to compute:
		// - near_j  - id of nearest bird
		// - ave_pos - average centroid of neighbor birds
		// - ave_vel - average velocity of neighbor birds	
		//
		float d = m_Accel.sim_scale;
		float d2 = d * d;
		float rd2 = (m_Accel.psmoothradius*m_Accel.psmoothradius) / d2;	
		int	nadj = (m_Accel.gridRes.z + 1)*m_Accel.gridRes.x + 1;
		uint j, cell;
		Vec3F posi, posj, dist;	
		Vec3F diri, dirj;
		Vec3F cdir;
		float dsq;
		float nearest, nearest_fwd;
	
		uint*		grid		=	m_Grid.bufUI(AGRID);
		uint*		gridcnt = m_Grid.bufUI(AGRIDCNT);
		uint*   fgc     = m_Grid.bufUI(FGCELL);

		Bird *bi, *bj;

		float ang, birdang;		
		
		// topological distance
		float sort_d_nbr[12];
		int sort_j_nbr[12];
		int sort_num = 0;
		sort_d_nbr[0] = 10^5;
		sort_j_nbr[0] = -1;		
		int k, m;

		m_centroid.Set(0,0,0);

		// for each bird..
		int numPoints = m_Params.num_birds;
		for (int i=0; i < numPoints; i++) {

			bi = (Bird*) m_Birds.GetElem( FBIRD, i);
			posi = bi->pos;
			m_centroid += posi;

			// pre-compute for efficiency
			diri = bi->vel;			diri.Normalize();
		
			// clear current bird info
			bi->ave_pos.Set(0,0,0);
			bi->ave_vel.Set(0,0,0);
			bi->near_j = -1;			
			bi->t_nbrs = 0;
			bi->r_nbrs = 0;

			nearest = rd2;
			nearest_fwd = rd2;

			sort_num = 0;

			// search neighbors
			int gc = m_Birds.bufUI(FGCELL)[i];
			if ( gc != GRID_UNDEF ) {

				gc -= nadj;

				for (int c=0; c < m_Accel.gridAdjCnt; c++) {
					cell = gc + m_Accel.gridAdj[c];
					int clast = m_Grid.bufUI(AGRIDOFF)[cell] + m_Grid.bufUI(AGRIDCNT)[cell];

					for ( int cndx = m_Grid.bufUI(AGRIDOFF)[cell]; cndx < clast; cndx++ ) {		

							// get next possible neighbor
							j = m_Grid.bufUI(AGRID)[cndx];
							if (i==j) continue;
							bj = (Bird*) m_Birds.GetElem( FBIRD, j );
							posj = bj->pos;

							dist = posi - posj;
							dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

							if ( dsq < rd2 ) {
								// neighbor is within radius..
								
								// confirm bird is within forward field-of-view							
								dirj = posj - posi; dirj.Normalize();
								birdang = diri.Dot (dirj);

								if ( birdang > m_Params.fovcos ) {

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

									// count bounary neighbors
									bi->r_nbrs++;

								}
							}
						}
					}
				}				
		
			// compute nearest and average among N (~7) topological neighbors
			for (k=0; k < sort_num; k++) {
				bj = (Bird*) m_Birds.GetElem( FBIRD, sort_j_nbr[k] );
				bi->ave_pos += bj->pos;
				bi->ave_vel += bj->vel;					
			}
			bi->near_j = sort_j_nbr[0];

			bi->t_nbrs = sort_num;
			if (sort_num > 0 ) {
				bi->ave_pos *= (1.0f / sort_num );
				bi->ave_vel *= (1.0f / sort_num );
			}

		}

		
		m_centroid *= (1.0f / numPoints);
	}
}



float circleDelta (float b, float a)
{	
	float d = b-a;
	d = (d > 180) ? d-360 : (d<-180) ? d+360 : d;
	//float q = fmod( fmin ( b-a, 360+b-a ), 180 );
	//printf ("%f %f\n", d, q);
	return d;
}

void Sample::DebugBird ( int id, std::string msg )
{
	int n;
	Bird* b;

	if (m_gpu) {
		#ifdef BUILD_CUDA
			m_Birds.Retrieve ( FBIRD );
			cuCtxSynchronize();
		#endif
	}
	
	for (n=0; n < m_Params.num_birds; n++) {
		b = (Bird*) m_Birds.GetElem (FBIRD, n);
		if (b->id == id) 
			break;
	}
	if (b->id == id) {
		printf ("-- BIRD: id %d, #%d (%s) -> %s\n", b->id, n, m_gpu ? "GPU" : "CPU", msg.c_str() );
		printf (" pos: %f, %f, %f\n", b->pos.x, b->pos.y, b->pos.z );
		printf (" vel: %f, %f, %f\n", b->vel.x, b->vel.y, b->vel.z );
		printf (" orient: %f, %f, %f, %f\n", b->orient.X, b->orient.Y, b->orient.Z, b->orient.W );
		printf (" target: %f, %f, %f\n", b->target.x, b->target.y, b->target.z);
		printf (" speed: %f\n", b->speed );
	}
}

void Sample::Advance ()
{
	if (m_gpu) {

		#ifdef BUILD_CUDA
			// Advance - GPU
			//
			int numPoints = m_Params.num_birds;
			void* args[4] = { &m_time, &m_Params.DT, &m_Accel.sim_scale, &numPoints };

			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_ADVANCE],  m_Accel.numBlocks, 1, 1, m_Accel.numThreads, 1, 1, 0, NULL, args, NULL), "Advance", "cuLaunch", "FUNC_ADVANCE", DEBUG_CUDA );

			// Retrieve birds from GPU for rendering & visualization
			m_Birds.Retrieve ( FBIRD );
		
			cuCtxSynchronize ();
		#endif

	} else {

		// Advance - CPU
		//
		Vec3F fwd, up, right, vaxis;
		Vec3F force, lift, drag, thrust, accel;
		Vec3F diri, dirj;
		Quaternion ctrl_pitch;
		float airflow, dynamic_pressure, aoa;

		float CL, L, dist, cd, vd;
		float pitch, yaw;
		Quaternion ctrlq, tq;
		Vec3F angs;
		Quaternion angvel;
		Bird *b, *bj, *bfwd;
		bool leader;
		float ave_yaw;

		int numPoints = m_Params.num_birds;

		Vec3F centroid (0,50,0);

		
		for (int n=0; n < numPoints; n++) {

			b = (Bird*) m_Birds.GetElem( FBIRD, n);

			b->clr.Set(0,0,0,0);	

			// Hoetzlein - Peripheral bird term
			// Turn isolated birds toward flock centroid
			float d = b->r_nbrs / m_Params.border_cnt;
			if ( d < 1 ) {
				b->clr.Set(1,.5,0, 1);	
				dirj = centroid - b->pos; dirj.Normalize();
				dirj *= b->orient.inverse();
				yaw = atan2( dirj.z, dirj.x )*RADtoDEG;
				pitch = asin( dirj.y )*RADtoDEG;
				b->target.z +=   yaw * m_Params.border_amt;
				b->target.y += pitch * m_Params.border_amt;
			}

			if ( b->r_nbrs > 0 ) {
				//--- Reynold's behaviors	
				// Rule 1. Avoidance - avoid nearest bird
				//			
				// 1a. Side neighbor avoidance
				if ( b->near_j != -1) {
					// get nearest bird
					bj = (Bird*) m_Birds.GetElem(0, b->near_j);
					dirj = bj->pos - b->pos;
					dist = dirj.Length();		  
			
					if ( dist < m_Params.safe_radius ) {	

						// Angular avoidance			
						dirj = (dirj/dist) * b->orient.inverse();													
						yaw = atan2( dirj.z, dirj.x )*RADtoDEG;
						pitch = asin( dirj.y )*RADtoDEG;		
						dist = fmax( 1.0f, fmin( dist*dist, 100.0f ));
						b->target.z -= yaw *		m_Params.avoid_angular_amt / dist;
						b->target.y -= pitch *  m_Params.avoid_angular_amt / dist;

						// Power adjust				
						L = (b->vel.Length() - bj->vel.Length()) * m_Params.avoid_power_amt;
						b->power = m_Params.avoid_power_ctr - L * L;				

					}			
				}
			
				if (b->power < m_Params.min_power) b->power = m_Params.min_power;
				if (b->power > m_Params.max_power) b->power = m_Params.max_power;	

				// Rule 2. Alignment - orient toward average direction		
				dirj = b->ave_vel;
				dirj.Normalize();
				dirj *= b->orient.inverse();		// using inverse orient for world-to-local xform
				yaw = atan2( dirj.z, dirj.x )*RADtoDEG;
				pitch = asin( dirj.y )*RADtoDEG;
				b->target.z += yaw   * m_Params.align_amt;
				b->target.y += pitch * m_Params.align_amt;		 

				// Rule 3. Cohesion - steer toward neighbor centroid
				dirj = b->ave_pos - b->pos;		// direction to ave nbrs
				dirj.Normalize();
				dirj *= b->orient.inverse();	// world-to-local xform		
				yaw = atan2( dirj.z, dirj.x )*RADtoDEG;  
				pitch = asin( dirj.y )*RADtoDEG;
				b->target.z += yaw   * m_Params.cohesion_amt;
				b->target.y += pitch * m_Params.cohesion_amt;		
		
			}
		}

		//--- Flight model
		//
		for (int n=0; n < numPoints; n++) {

			b = (Bird*) m_Birds.GetElem( FBIRD, n);

			#ifdef DEBUG_BIRD
				if (b->id == DEBUG_BIRD) {
					printf ("---- ADVANCE START (CPU), id %d, #%d\n", b->id, n );
					printf (" orient:  %f, %f, %f, %f\n", b->orient.X, b->orient.Y, b->orient.Z, b->orient.W );		
					printf (" target:  %f, %f, %f\n", b->target.x, b->target.y, b->target.z );		
				}
			#endif
			
			// Body orientation
			fwd = Vec3F(1,0,0) * b->orient;			// X-axis is body forward
			up  = Vec3F(0,1,0) * b->orient;			// Y-axis is body up
			right = Vec3F(0,0,1) * b->orient;		// Z-axis is body right

			// Direction of motion
			b->speed = b->vel.Length();
			vaxis = b->vel / b->speed;	
			if ( b->speed < m_Params.min_speed ) b->speed = m_Params.min_speed;				// birds dont go in reverse
			if ( b->speed > m_Params.max_speed ) b->speed = m_Params.max_speed;
			if ( b->speed==0) vaxis = fwd;
			
			b->orient.toEuler ( angs );			

			// Target corrections
			angs.z = fmod (angs.z, 180.0 );												
			b->target.z = fmod ( b->target.z, 180 );										// yaw -180/180
			b->target.x = circleDelta(b->target.z, angs.z) * 0.5;				// banking
			b->target.y *= m_Params.pitch_decay;																				// level out
			if ( b->target.y < m_Params.pitch_min ) b->target.y = m_Params.pitch_min;
			if ( b->target.y > m_Params.pitch_max ) b->target.y = m_Params.pitch_max;
			if ( fabs(b->target.y) < 0.0001) b->target.y = 0;

			// Compute angular acceleration
			// - as difference between current direction and desired direction
			b->ang_accel.x = (b->target.x - angs.x);
			b->ang_accel.y = (b->target.y - angs.y);
			b->ang_accel.z = circleDelta(b->target.z, angs.z);
		
			// Roll - Control input
			// - orient the body by roll
			ctrlq.fromAngleAxis ( b->ang_accel.x * m_Params.reaction_delay, fwd );
			b->orient *= ctrlq;	b->orient.normalize();

			// Pitch & Yaw - Control inputs
			// - apply 'torque' by rotating the velocity vector based on pitch & yaw inputs				
			ctrlq.fromAngleAxis ( b->ang_accel.z * m_Params.reaction_delay, up * -1.f );
			vaxis *= ctrlq; vaxis.Normalize();	
			ctrlq.fromAngleAxis ( b->ang_accel.y * m_Params.reaction_delay, right );
			vaxis *= ctrlq; vaxis.Normalize();

			// Adjust velocity vector
			b->vel = vaxis * b->speed;
			force = 0;

			// Dynamic pressure		
			airflow = b->speed + m_Params.wind.Dot ( fwd*-1.0f );		// airflow = air over wing due to speed + external wind			
			float dynamic_pressure = 0.5f * m_Params.air_density * airflow * airflow;

			// Lift force
			aoa = acos( fwd.Dot( vaxis ) )*RADtoDEG + 1;		// angle-of-attack = angle between velocity and body forward		
 			if (isnan(aoa)) aoa = 1;
			// CL = sin(aoa * 0.2) = coeff of lift, approximate CL curve with sin
			L = sin( aoa * 0.2) * dynamic_pressure * m_Params.lift_factor * 0.5;		// lift equation. L = CL (1/2 p v^2) A
			lift = up * L;
			force += lift;	

			// Drag force	
			drag = vaxis * dynamic_pressure * m_Params.drag_factor  * -1.0f;			// drag equation. D = Cd (1/2 p v^2) A
			force += drag; 

			// Thrust force
			thrust = fwd * b->power;
			force += thrust;
	
			// Integrate position		
			accel = force / m_Params.mass;				// body forces	
			accel += m_Params.gravity;						// gravity
			accel += m_Params.wind * m_Params.air_density * m_Params.front_area;		// wind force. Fw = w^2 p * A, where w=wind speed, p=air density, A=frontal area
	
			b->pos += b->vel * m_Params.DT;

			// Boundaries
			if ( b->pos.x < m_Accel.bound_min.x ) b->pos.x = m_Accel.bound_max.x;
			if ( b->pos.x > m_Accel.bound_max.x ) b->pos.x = m_Accel.bound_min.x;
			if ( b->pos.z < m_Accel.bound_min.z ) b->pos.z = m_Accel.bound_max.z;
			if ( b->pos.z > m_Accel.bound_max.z ) b->pos.z = m_Accel.bound_min.z;			  

			// Ground avoidance
			L = b->pos.y - m_Accel.bound_min.y;
			if ( L < m_Params.bound_soften ) {			
				L = (m_Params.bound_soften - L) / m_Params.bound_soften;
				b->target.y += L * m_Params.avoid_ground_amt;			
				// power up so we have enough lift to avoid the ground
				b->power = m_Params.avoid_ground_power;
			} 
		
			// Ceiling avoidance
			L = m_Accel.bound_max.y - b->pos.y;
			if ( L < m_Params.bound_soften  ) {	
				L = (m_Params.bound_soften - L) / m_Params.bound_soften;
				b->target.y -= L * m_Params.avoid_ceil_amt; 						
			} 

			// Ground condition
			if (b->pos.y <= 0.00001 ) { 
				// Ground forces
				b->pos.y = 0; b->vel.y = 0; 
				b->accel += Vec3F(0,9.8,0);	// ground force (upward)
				b->vel *= 0.9999;				// ground friction
				b->orient.fromDirectionAndRoll ( Vec3F(fwd.x, 0, fwd.z), 0 );	// zero pitch & roll			
			} 
	
			// Integrate velocity
			b->vel += accel * m_Params.DT;		

			vaxis = b->vel;	vaxis.Normalize ();

			// Update Orientation
			// Directional stability: airplane will typically reorient toward the velocity vector
			//  see: https://github.com/ramakarl/Flightsim
			// this is an assumption yet much simpler/faster than integrating body orientation
			// this way we dont need torque, angular vel, or rotational inertia.
			// stalls are possible but not flat spins or 3D flying		
			angvel.fromRotationFromTo ( fwd, vaxis, m_Params.dynamic_stability );
			if ( !isnan(angvel.X) ) {
				b->orient *= angvel;
				b->orient.normalize();			
			}

			#ifdef DEBUG_BIRD
				if (b->id == DEBUG_BIRD) {
					printf ("---- ADVANCE (CPU), id %d, #d\n", b->id, n );
					printf (" speed:   %f\n", b->speed );
					printf (" airflow: %f\n", airflow );
					printf (" orients: %f, %f, %f, %f\n", b->orient.X, b->orient.Y, b->orient.Z, b->orient.W );		
					printf (" angs:    %f, %f, %f\n", angs.x, angs.y, angs.z );		
					printf (" target:  %f, %f, %f\n", b->target.x, b->target.y, b->target.z );								
				}
			#endif
		}
	}

	
}

void Sample::SelectBird (float x, float y)
{
	// camera ray
	Vec3F rpos = m_cam->getPos ();
	Vec3F rdir = m_cam->inverseRay ( x, y, getWidth(), getHeight() );
	Vec3F q;
	float dist;
	int best_id;
	float best_dist;

	Bird* b;

	best_id = -1;
	best_dist = 10^5;

	// find the bird nearest to camera ray
	for (int i=0; i < m_Params.num_birds; i++) {		
		b = (Bird*) m_Birds.GetElem( FBIRD, i );

		q = projectPointLine( b->pos, rpos, rpos+rdir );
		dist = (b->pos - q).Length();
		if ( dist < best_dist ) {
			best_id = b->id;
			best_dist = dist;
		}
	}

	// set as selection
	// *note* due to GPU sort, the array index of selected bird 
	// may change frame-to-frame. therefore, selection is the bird ID.
	if ( best_dist < 5 ) {
		m_bird_sel = best_id;
	} else {
		m_bird_sel = -1;
	}
}

void Sample::InitGraphs ()
{
	graph_t g;
	for (int i=0; i < GRAPH_MAX; i++) {
		g.x = 0;
		memset ( &g.y[0], 0, 2048 * sizeof(float) );
		g.clr = Vec4F(1,1,1,1);
		m_graph.push_back ( g );
	}
	m_graph[ GRAPH_BANK ].clr  = Vec4F(1,0,0,1 );				// red
	m_graph[ GRAPH_PITCH ].clr = Vec4F(1,0.5,0,1 );			// orange
	m_graph[ GRAPH_VEL ].clr   = Vec4F(0,1,0,1 );				// green
	m_graph[ GRAPH_ACCEL ].clr = Vec4F(0,0,1,1 );				// blue

}

void Sample::VisualizeSelectedBird ()
{
	// selection is bird ID

	// avoid the work is nothing selected
	if (m_bird_sel==-1) return;

	// search for the index of this bird
	m_vis.clear ();	
	Bird* b;
	int ndx = -1;
	for (int i=0; i < m_Params.num_birds; i++) {
		b = (Bird*) m_Birds.GetElem ( FBIRD, i );
		if ( b->id == m_bird_sel )  {
			ndx = i;
			break;
		}
	}

	if (ndx != -1 ) {

		m_bird_ndx = ndx;

		// visualize bird (green)
		m_vis.push_back ( vis_t( b->pos, 1.1f, Vec4F(0,1,0,1) ) );

		// visualize neighborhood radius (yellow)
		m_vis.push_back ( vis_t( b->pos, m_Accel.psmoothradius, Vec4F(1,1,0,1) ) );

		// graphs
		Vec3F angs;
		b->orient.toEuler ( angs );				
		if (++m_graph[0].x >= 2048) m_graph[0].x = 0;
		int x = m_graph[0].x;
		m_graph[GRAPH_BANK].y[x] = angs.x;					// banking
		m_graph[GRAPH_PITCH].y[x] = angs.y;					// pitch
		m_graph[GRAPH_VEL].y[x] = b->vel.Length();	// velocity

		// visulize neighbors		
		if (m_gpu) {
			#ifdef BUILD_CUDA
				m_Birds.Retrieve ( FGCELL );
				m_Grid.RetrieveAll ();
				cuCtxSynchronize();
			#endif
		}
		int gc = m_Birds.bufUI(FGCELL)[ ndx ];
		if ( gc != GRID_UNDEF ) {			
			Bird* bj;
			float dsq, ave_dist = 0;
			Vec3F dist;
			uint j, cell, ncnt = 0;			

			// find neighbors
			float rd2 = (m_Accel.psmoothradius*m_Accel.psmoothradius) / (m_Accel.sim_scale * m_Accel.sim_scale);
			gc -= (m_Accel.gridRes.z + 1)*m_Accel.gridRes.x + 1;
			for (int c=0; c < m_Accel.gridAdjCnt; c++) {
				cell = gc + m_Accel.gridAdj[c];
				int clast = m_Grid.bufUI(AGRIDOFF)[cell] + m_Grid.bufUI(AGRIDCNT)[cell];
				for ( int cndx = m_Grid.bufUI(AGRIDOFF)[cell]; cndx < clast; cndx++ ) {		
						// get next possible neighbor
						j = m_Grid.bufUI(AGRID)[cndx];
						if (j==ndx) continue;
						bj = (Bird*) m_Birds.GetElem ( FBIRD, j );
						dist = b->pos - bj->pos;
						dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
						if ( dsq < rd2 ) {							
							ave_dist += sqrt( dsq );
							ncnt++;
							m_vis.push_back ( vis_t( bj->pos, 0.5f, Vec4F(1,1,0,1) ) );		// neighbor birds (yellow)
						}
				}
			}
			if ( ncnt > 0) {
				ave_dist /= ncnt;
				printf ( "ave dist: %f\n", ave_dist );
			}
		}					

	}
}


// Run
// run a single time step
//
void Sample::Run ()
{
	// PERF_PUSH ( "Run" );

	TimeX t1, t2;
	t1.SetTimeNSec();

	#ifdef DEBUG_BIRD
		DebugBird ( DEBUG_BIRD, "Start" );
	#endif

	//--- Insert birds into acceleration grid
	InsertIntoGrid ();

	//--- Prefix scan for accel grid
	PrefixSumGrid ();

	//--- Find neighbors
	FindNeighbors ();

	//--- Advance birds with behaviors & flight model
	Advance ();			

	#ifdef DEBUG_BIRD
		DebugBird ( 7, "Post-Advance" );
	#endif

	// computation timing
	t2.SetTimeNSec();
	float msec = t2.GetElapsedMSec( t1 );
	printf ( "Run: %f msec/step, %2.2f%% real-time\n", msec, (m_Params.DT*1000.0)*100 / msec );

	// PERF_POP();

	m_time += m_Params.DT;
}


void Sample::DrawAccelGrid ()
{
	Vec3F r,a,b;
	float v;

	uint* gc = (uint*) m_Grid.bufUI(AGRIDCNT);

	for (r.y=0; r.y < m_Accel.gridRes.y; r.y++) {
		for (r.z=0; r.z < m_Accel.gridRes.z; r.z++) {
			for (r.x=0; r.x < m_Accel.gridRes.x; r.x++) {
				
				a = m_Accel.gridMin + r / m_Accel.gridDelta;
				b = a + (Vec3F(0.99f,0.99f,0.99f) / m_Accel.gridDelta );								

				v = fmin(1.0, float(*gc)/10.0f);

				drawBox3D ( a, b, Vec4F(v, 1-v, 1-v, 0.02 + v) );

				gc++;
			}
		}
	}

}

void Sample::CameraToBird ( int n )
{
	Bird* b = (Bird*) m_Birds.GetElem(0, n);

	m_cam->SetOrbit ( m_cam->getAng(), b->pos, m_cam->getOrbitDist(), m_cam->getDolly() );
}

void Sample::CameraToCentroid ( )
{
	m_centroid = 0;
	for (int n=0; n < m_Params.num_birds; n++) {
		Bird* b = (Bird*) m_Birds.GetElem( FBIRD, n );
		m_centroid += b->pos;
	}
	m_centroid *= (1.0 / m_Params.num_birds);

	if (!m_cam_adjust) {
		m_cam->setDirection ( m_cam->getPos(), m_centroid, 0 );
	}
}

void Sample::CameraToCockpit(int n )
{
	Bird* b = (Bird*) m_Birds.GetElem(0, n);

	// View direction	
	Vec3F fwd = b->vel; fwd.Normalize();
	Vec3F angs;
	b->orient.toEuler ( angs );

	m_cam_fwd = m_cam_fwd * 0.99f + fwd * 0.01f;
	m_cam_fwd.Normalize();

	// Set eye level above centerline
	Vec3F p = b->pos + Vec3F(0,2,0);	  
	
	m_cam->setDirection ( p, p + m_cam_fwd, -angs.x );
}


bool Sample::init ()
{
	int w = getWidth(), h = getHeight();			// window width &f height

	appSetVSync( false );

	PERF_INIT ( 64, false, true, false, 0, "");
	
	m_run = true;	
	m_cockpit_view = false;
	m_draw_sphere = false;
	m_draw_grid = false;
	m_draw_vis = true;
	m_cam_mode = 0;
	m_time = 0;
	m_rnd.seed (12);
	
	// enable GPU if cuda available
	#ifdef BUILD_CUDA
		m_gpu = true;
  #else
	  m_gpu = false;
	#endif

	// override. use CPU 
	// m_gpu = false;


	m_kernels_loaded = false;

	m_bird_sel = -1;
	
	addSearchPath ( ASSET_PATH );	

	init2D ( "arial" );


	InitGraphs ();

	// [optional Start GPU
	if (m_gpu) {
		#ifdef BUILD_CUDA
			cuStart ( DEV_FIRST, 0, m_dev, m_ctx, 0, true );
		#endif
	}

	// Initialize flock simulation
	// 
  
	DefaultParams ();

	
	int num_birds = 2000;

	Reset ( num_birds );


	// Camera
	m_cam = new Camera3D;
	m_cam->setFov ( 70 );
	m_cam->setNearFar ( 1.0, 100000 );
	m_cam->SetOrbit ( Vec3F(-30,30,0), Vec3F(0,50,0), 300, 1 );

	return true;
}

void Sample::drawBackground ()
{
	int w = getWidth(), h = getHeight();
	
	if ( m_draw_vis ) {
		// black background for vis
		drawFill ( Vec2F(0,0), Vec2F(w,h), Vec4F(.4,.4,.4,1) );
	} else {
		// realistic sky
		//drawFill ( Vec2F(0,0), Vec2F(w,h), Vec4F(1,1,1,1) );
		drawGradient ( Vec2F(0,0), Vec2F(w,h), Vec4F(.6,.7,.8,1), Vec4F(.6,.6,.8,1), Vec4F(1,1,.9,1), Vec4F(1,1,.9,1) );
	}
}

void Sample::display ()
{	
	char msg[2048];
	Vec3F x,y,z;
	Vec3F pnt;
	Vec4F clr;
	int w = getWidth();
	int h = getHeight();

	Bird* b;

	// Advance simulation
	if (m_run) { 		

		for (int i=0; i < m_Params.steps; i++)
			Run ();
	}	

	// Graph & visualize selected bird
	VisualizeSelectedBird ();


	// CameraToCentroid ();

	/*if (m_cockpit_view) {
		CameraToCockpit ( m_bird_sel);
	} else {
		CameraToBird ( m_bird_sel );
  }*/

	if ( m_draw_vis ) {
		glClearColor(1,1,1,1);
	} else {
		glClearColor(.8,.8,.9,1);
	}
	clearGL();

	start2D( w, h);
		drawBackground ();
	end2D();

	start3D(m_cam);		

		setLight3D ( Vec3F(0, 400, 0), Vec4F(0.1, 0.1, 0.1, 1) );	
		setMaterial ( Vec3F(0,0,0), Vec3F(0,0,0), Vec3F(.2,.2,.2), 40, 1.0 );

		// Draw selected bird 
		if (m_bird_sel != -1) {
			
			// draw visualization elements 
			// this includes:
			// - selected bird (green)
			// - neighobr birds (yellow)
			// - nearest bird (red)
			Vec3F cn, p;
			for (int k=0; k < m_vis.size(); k++) {				
				drawCircle3D ( m_vis[k].pos, m_cam->getPos(), m_vis[k].radius, m_vis[k].clr );
			}			
		}		

		// Draw acceleration grid
		if (m_draw_grid) {
			drawBox3D ( m_Accel.bound_min, m_Accel.bound_max, Vec4F(0,1,1,0.5) );
			DrawAccelGrid ();
		}

		// Draw ground plane
		if ( m_draw_vis ) {
			//drawLine3D ( Vec3F(0,0,0), Vec3F(100,0,0), Vec4F(1,0,0,1));
			//drawLine3D ( Vec3F(0,0,0), Vec3F(  0,0,100), Vec4F(0,0,1,1));
			//drawGrid( Vec4F(0.1,0.1,0.1, 1) );
		}

		for (int n=0; n < m_Birds.GetNumElem(0); n++) {

			b = (Bird*) m_Birds.GetElem(0, n);

			if ( m_draw_vis ) {
				
				// visualize velocity
				float v = (b->vel.Length() - m_Params.min_speed) / (m_Params.max_speed - m_Params.min_speed);
				//float v2 = (b->power - 2) / 2.0;
				float v2 = b->ang_accel.Length() / 24.0;

				if (b->clr.w==0) {
					drawLine3D ( b->pos,		b->pos + (b->vel*0.1f),	Vec4F(0, 1-v2, v2,1) );
				} else {
					drawLine3D ( b->pos,		b->pos + (b->vel*0.1f),	b->clr );
				}

			} else {
				// bird dart
				x = Vec3F(1,0,0) * b->orient;
				y = Vec3F(0,1,0) * b->orient;
				z = Vec3F(0,0,1) * b->orient;
				Vec3F p,q,r,t;
				p = b->pos - z * 0.3f;   // wingspan = 40 cm = 0.2m (per wing)
				q = b->pos + z * 0.3f;
				r = b->pos + x * 0.4f;   // length = 22 cm = 0.22m
				t = y;				
				drawTri3D ( p, q, r, t, Vec4F(1,1,1,1) );
			}
		
		}
	end3D(); 

	start2D ( w, h );
	
		Vec4F tc (1,1,1,1);
		sprintf ( msg, "t=%4.3f sec", m_time );
		setTextSz ( 24, 0 );
	  drawText ( Vec2F(10, 10), "hello world", tc );
		
		if ( m_bird_sel != -1) {
			

			Bird* bsel = (Bird*) m_Birds.GetElem ( FBIRD, m_bird_ndx );	// use index here
			sprintf ( msg, "x: %f y: %f\n", getX(), getY() );
			drawText ( Vec2F(10, 10), msg, tc );
			sprintf ( msg, "pos: %f %f %f\n", bsel->pos.x, bsel->pos.y, bsel->pos.z );
			drawText ( Vec2F(10, 30), msg, tc );
			sprintf ( msg, "vel: %f %f %f = %f\n", bsel->vel.x, bsel->vel.y, bsel->vel.z, bsel->vel.Length() );
			drawText ( Vec2F(10, 50), msg, tc );
			sprintf ( msg, "power: %f\n", bsel->power );
			drawText ( Vec2F(10, 70), msg, tc );
			
		
			// Graph selected bird 
			if (m_bird_sel != -1) {			
				Vec2F a,b;
				// draw graphs
				drawRect ( Vec2F(0, 20), Vec2F(getWidth(), 420.f), Vec4F(.5,.5,.5,1 ));
				drawLine ( Vec2F(0, 210), Vec2F(getWidth(), 210.f), Vec4F(.5,.5,.5,1 ));
				for (int k=0; k < m_graph.size(); k++) {
					b = Vec3F( 0, 210 - m_graph[k].y[0], 0);
					for (int x=0; x < 2084; x++) {
						a = Vec3F( x, 210 - m_graph[k].y[x], 0);
						drawLine ( a, b, m_graph[k].clr );
						b = a;
					}
				}
			}
		}

	end2D(); 

	drawAll ();
	
	appPostRedisplay();								// Post redisplay since simulation is continuous
}


void Sample::mouse(AppEnum button, AppEnum state, int mods, int x, int y)
{
	int w = getWidth(), h = getHeight();				// window width & height

	mouse_down = (state == AppEnum::BUTTON_PRESS) ? button : -1;

	if (mouse_down == AppEnum::BUTTON_LEFT) {
		SelectBird ( x, y );
	}
}


void Sample::motion (AppEnum button, int x, int y, int dx, int dy) 
{
	// Get camera for scene
	bool shift = (getMods() & KMOD_SHIFT);		// Shift-key to modify light
	float fine = 0.5f;
	Vec3F dang; 

	m_cam_adjust = false;

	switch ( mouse_down ) {	
	case AppEnum::BUTTON_LEFT:  {	
	
		} break;

	case AppEnum::BUTTON_MIDDLE: {
		// Adjust target pos				
		float zoom = (m_cam->getOrbitDist() - m_cam->getDolly()) * 0.0003f;
		m_cam->moveRelative ( float(dx) * zoom, float(-dy) * zoom, 0 );	
		m_cam_adjust = true;
		} break; 

	case AppEnum::BUTTON_RIGHT: {
		// Adjust orbit angles
		Vec3F angs = m_cam->getAng();
		if (m_draw_vis) { 			
			angs.x += dx*0.2f;
			angs.y -= dy*0.2f;				
			m_cam->SetOrbit ( angs, m_cam->getToPos(), m_cam->getOrbitDist(), m_cam->getDolly() );			
		} else {		
			angs.x += dx*0.02f;
			angs.y -= dy*0.02f;				
			m_cam->setAngles ( angs.x, angs.y, angs.z );		}
			m_cam_adjust = true;
		} break;	
	}
}

void Sample::mousewheel(int delta)
{
	// Adjust zoom
	float zoomamt = 1.0;
	float dist = m_cam->getOrbitDist();
	float dolly = m_cam->getDolly();
	float zoom = (dist - dolly) * 0.0005f;
	dist -= delta * zoom * zoomamt;
	
	m_cam->SetOrbit(m_cam->getAng(), m_cam->getToPos(), dist, dolly);		
	m_cam_adjust = true;
}




void Sample::keyboard(int keycode, AppEnum action, int mods, int x, int y)
{
	if (action == AppEnum::BUTTON_RELEASE) 
		return;

	switch ( keycode ) {
	case 'v': 
		m_draw_vis = !m_draw_vis;	
		break;
	case 's': m_draw_sphere = !m_draw_sphere; break;
	case 'g': m_draw_grid = !m_draw_grid; break;
  case 'c': 		
		m_cockpit_view = !m_cockpit_view; 
		//m_cam_orient = 
		break;
	case 'r': Reset( m_Params.num_birds ); break;
	case ' ':	m_run = !m_run;	break;	
	case 'z': 
		m_bird_sel--; 
		if (m_bird_sel < 0) m_bird_sel = 0; 
		break;
	case 'x':
		m_bird_sel++; 
		if (m_bird_sel > m_Birds.GetNumElem(0)) m_bird_sel = m_Birds.GetNumElem(0)-1;		
		break;
	};
	printf ( "%d \n", m_bird_sel );
}

void Sample::reshape (int w, int h)
{
	glViewport ( 0, 0, w, h );
	setview2D ( w, h );

	m_cam->setAspect(float(w) / float(h));
	m_cam->SetOrbit(m_cam->getAng(), m_cam->getToPos(), m_cam->getOrbitDist(), m_cam->getDolly());	
		
	appPostRedisplay();	
}

void Sample::startup ()
{
	int w = 1900, h = 1000;
	appStart ( "Flock v2 (c) Rama Karl 2023, MIT license", "Flock v2", w, h, 4, 2, 16, false );	
}

void Sample::shutdown()
{
}


