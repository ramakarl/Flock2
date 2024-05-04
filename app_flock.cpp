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

#include <fstream>
#include <iostream>

using namespace std;

#define DEBUG_CUDA		true	
//#define DEBUG_BIRD		7

#include "gxlib.h"			// low-level render
#include "g2lib.h"			// gui system
using namespace glib;

// Bird structures
//
#include "flock_types.h"

// Visualization
//
struct vis_t {
	vis_t( Vec3F p, float r, Vec4F c, std::string t) {pos=p; radius=r; clr=c; txt=t;}
	Vec3F				pos;
	float				radius;
	Vec4F				clr;
	std::string txt;
};
struct graph_t {
	int			x;
	float		y[2048];
	Vec2F		scal;
	Vec4F		clr;
};
#define GRAPH_BANK		0
#define GRAPH_PITCH		1
#define GRAPH_VEL			2
#define GRAPH_ACCEL		3
#define GRAPH_MAX			4

#define NUM_BIRDS		10000
#define MAX_BIRDS		65535

#define SAMPLES			16384
#define PLOT_RESX		2048
#define PLOT_RESY		800

// FFTW Analysis
#ifdef USE_FFTW
	#include <fftw3.3/fftw3.h>	

#endif

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

	// Simulation
	Bird*			AddBird ( Vec3F pos, Vec3F vel, Vec3F target, float power );
	void			DefaultParams ();
	void			Reset (int num_bird, int num_pred);	
	void			Run ();
	void			FindNeighbors ();	
	void			AdvanceByOrientation ();	
	void			AdvanceByVectors  ();	
	void			UpdateFlockData ();
	void			OutputPlot ( int what, int frame );
	void			OutputPointCloudFiles (int frame);
	void			OutputFFTW ( int frame );
	void			StartNextRun ();
	
	// Predators
	Predator*	AddPredator(Vec3F pos, Vec3F vel, Vec3F target, float power);		//predators
	void 			Advance_pred();		
	void			TrackBird();			
	
	// Rendering
	void			SelectBird (float x, float y);
	void			Graph ( int id, float y, Vec4F clr, Vec2F scal );
	void			VisualizeSelectedBird ();	
	void			DebugBird ( int id, std::string msg );
	void			CameraToBird ( int b );
	void			CameraToCockpit( int b );
	void			CameraToCentroid ();
	void			drawBackground();
	void			drawGrid( Vec4F clr );

	// Acceleration
	void			InitializeGrid ();
	void			InsertIntoGrid ();
	void			PrefixSumGrid ();
	void			DrawAccelGrid ();	

	void			transitionPredState(int centroidReached, predState& currentState);
	int				centroidReached;

	// Birds
	DataX			m_Birds;	
	DataX			m_BirdsTmp;			
	DataX			m_Grid;				// spatial grid accel
	Accel			m_Accel;			// accel parameters
	Params		m_Params;			// simulation params
	Flock			m_Flock;			// flock data
	
	// Sim setup
	float			m_time;
	int				m_frame;
	int				m_start_frame;
	int				m_end_frame;
	int				m_rec_start, m_rec_step;

	// Predators
	DataX			m_Predators;
	Vec3F			m_predcentroid;	//-------

	Mersenne		m_rnd;	
	
	// Rendering
	bool			m_running;
	int				m_cam_mode;
	bool			m_cam_adjust;
	Camera3D*		m_cam;
	Vec3F			m_cam_fwd;
	int				mouse_down;
	int				m_bird_sel, m_bird_ndx;
	int				m_method; 
	bool			m_cockpit_view;
	bool			m_draw_sphere;
	bool			m_draw_grid;
	bool			m_draw_plot;
	int				m_draw_vis;
	bool			m_gpu, m_kernels_loaded;
	int				bird_index;		
	float			closest_bird;	
	int				bird_count = 0;
	int				runcount = 0;
	
	// Stats - Output files	
	FILE*			m_runs_outfile;

	// Stats - Image plots
	ImageX		m_plot[2];

	// Stats - Bird vis, graphs, lines
	std::vector< vis_t >  m_vis;
	std::vector< graph_t> m_graph;
	std::vector< Vec4F >	m_lines;

	// Stats - Frequency analysis
	#ifdef USE_FFTW
		double*				m_samples;
		double*				m_fftw_in;
		int						m_fftw_N;
		fftw_plan			m_fftw_plan;
		fftw_complex* m_fftw_out;		
		float					m_fftw_energy[32767];
		float					m_freq_grp[32767][4];
		float					m_freq_gmin[4];
		float					m_freq_gmax[4];
		float					m_fftw_s1[32767];
		float					m_fftw_s2[32767];
		int						m_peak_cnt;
		float					m_peak_ave;
		float					m_peak_max;
	#endif			

	// Experiment setup
	int				m_run;
	int				m_num_run;
	Vec3F			m_val;


	// CUDA / GPU
	#ifdef BUILD_CUDA
		void			LoadKernel ( int fid, std::string func );
		void			LoadAllKernels ();

		CUcontext		m_ctx;
		CUdevice		m_dev; 
		CUdeviceptr	m_cuAccel;
		CUdeviceptr	m_cuParam;
		CUdeviceptr	m_cuFlock;
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
		//LoadKernel ( KERNEL_FIND_NBRS,			"findNeighbors" );
		LoadKernel ( KERNEL_ADVANCE_ORIENT,	"advanceByOrientation" );
		LoadKernel ( KERNEL_ADVANCE_VECTORS,"advanceByVectors" );
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
	
	m_Birds.SetElem (FBIRD, ndx, &b );

	return (Bird*) m_Birds.GetElem( FBIRD, ndx);
}

Predator* Sample::AddPredator(Vec3F pos, Vec3F vel, Vec3F target, float power)	// **** copy of "AddBird"
{
	Vec3F dir, angs;

	int ndx = m_Predators.AddElem(FPREDATOR);

	Predator p;
	p.id = ndx;
	p.pos = pos;
	p.vel = vel;
	p.target = target;
	p.power = power;
	p.pitch_adv = 0;
	p.accel.Set(0, 0, 0);

	dir = p.vel; dir.Normalize();
	p.orient.fromDirectionAndUp(dir, Vec3F(0, 1, 0));
	p.orient.normalize();
	p.orient.toEuler(angs);

	p.currentState = HOVER;

	m_Predators.SetElem(FPREDATOR, ndx, &p);
	//printf("Predator is added at: %f, %f, %f \n", p.pos.x, p.pos.y, p.pos.z);

	return (Predator*)m_Predators.GetElem(FPREDATOR, ndx);
} // ****

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
	m_Params.steps = 2;
	m_Params.DT = 0.010;							// timestep (sec), .005 = 5 msec = 200 hz
	
	m_Params.mass =	0.08;							// bird mass (kg) - starling
	m_Params.power = 0.3873;							// 100% power (in joules)
	m_Params.min_speed = 5;						// min speed (m/s)		// Demsar2014
	m_Params.max_speed = 18;					// max speed (m/s)		// Demsar2014		
	m_Params.min_power = -20;					// min power (N)
	m_Params.max_power = 20;					// max power (N)
	m_Params.wind =	Vec3F(0,0,0);			// wind direction & strength
	m_Params.fov = 290;								// bird field-of-view (degrees), max = 360 deg (180 left & right)	
	
	// social factors
	m_Params.boundary_cnt = 160;						// border width (# birds)	
	m_Params.boundary_amt = 0.60f;				// border steering amount (keep <0.1)	
	
	//-- disable border
	//	m_Params.border_cnt = 0;
	//		m_Params.border_amt = 0.0f;

	m_Params.avoid_angular_amt= 0.06f;	// bird angular avoidance amount
	m_Params.avoid_power_amt =	0.00f;	// power avoidance amount (N)
	m_Params.avoid_power_ctr =	3;			// power avoidance center (N)	
	m_Params.align_amt = 0.500f;				// bird alignment amount
	m_Params.cohesion_amt =	0.002f;			// bird cohesion amount

	// flight parameters
	m_Params.wing_area = 0.0224;
	m_Params.lift_factor = 0.5714;			// lift factor
	m_Params.drag_factor = 0.1731;			// drag factor 
	m_Params.safe_radius = 2.0;					// radius of avoidance (m)
	m_Params.pitch_decay = 0.95;				// pitch decay (return to level flight)
	m_Params.pitch_min = -40;						// min pitch (degrees)
	m_Params.pitch_max = 40;						// max pitch (degrees)	
	m_Params.reaction_speed = 4500;			// reaction speed (millisec)
	m_Params.dynamic_stability = 0.5f;	// dyanmic stability factor
	m_Params.air_density = 1.225;				// air density (kg/m^3)
	m_Params.gravity = Vec3F(0, -9.8, 0);		// gravity (m/s^2)
	m_Params.front_area = 0.1f;					// section area of bird into wind
	m_Params.bound_soften = 20;					// ground detection range
	m_Params.avoid_ground_power = 3;			// ground avoid power setting 
	m_Params.avoid_ground_amt = 0.5f;			// ground avoid strength
	m_Params.avoid_ceil_amt = 0.1f;				// ceiling avoid strength

	// good orientation waves: reaction_delay=.002, dyn_stability=0.5

	// Predator
	m_Params.pred_radius = 10.0;					// detection radius of predator for birds
	m_Params.pred_mass = 0.8;	
	m_Params.max_predspeed = 22;				// m/s
	m_Params.min_predspeed = 8.8;				// m/s	
	//m_Params.pred_flee_speed = m_Params.max_speed;	// bird speed to get away from predator
	m_Params.avoid_pred_angular_amt = 0.04f;			// bird angular avoidance amount w.r.t. predator
	m_Params.avoid_pred_power_amt = 0.04f;				// power avoidance amount (N) w.r.t. predator
	m_Params.avoid_pred_power_ctr = 3;					// power avoidance center (N) w.r.t. predator

	m_Params.fov_pred = 120; // degrees
	m_Params.fovcos_pred = cos(m_Params.fov_pred * DEGtoRAD);

	// testing level flight	- no social factors, only balanced flight
	/* m_Params.align_amt = 0.0f;	
	m_Params.cohesion_amt =	0.0f;
	m_Params.avoid_angular_amt = 0.0f;
	m_Params.border_amt = 0.0f;   */

	// Reynold's Classic model		
	
	m_Params.reynolds_avoidance = 0.5;
	m_Params.reynolds_alignment = 1.0;
	m_Params.reynolds_cohesion =  0.2;
	
	
}


void Sample::Reset (int num, int num_pred )
{
	Vec3F pos, vel;
	float h;
	int grp;
	Bird* b;
	Predator* p;

	if ( num > MAX_BIRDS ) {
			printf ("ERROR: Maximum bird limit.\n" );
	}

	// Global flock variables
	//
	m_Params.num_birds = num;
	m_Params.num_predators = num_pred; 	// ****

	// Calculated params
	m_Params.fovcos = cos ( m_Params.fov * 0.5 * DEGtoRAD );
	
	// Initialized bird memory
	//
	int numPoints = m_Params.num_birds;
	int numPoints_pred = m_Params.num_predators; // ****
	uchar usage = (m_gpu) ? DT_CPU | DT_CUMEM : DT_CPU;

	m_Birds.DeleteAllBuffers ();
	m_Birds.AddBuffer ( FBIRD,  "bird",		sizeof(Bird),	numPoints, usage );
	m_Birds.AddBuffer ( FGCELL, "gcell",	sizeof(uint),	numPoints, usage );
	m_Birds.AddBuffer ( FGNDX,  "gndx",		sizeof(uint),	numPoints, usage );	

	dbgprintf ( "Bird sz: %d\n", sizeof(Bird)) ;

	// -------- PREDATOR -----
	m_Predators.DeleteAllBuffers();
	m_Predators.AddBuffer(FPREDATOR, "predator", sizeof(Predator), numPoints_pred, usage);	

	// Add birds
	//
	for (int n=0; n < numPoints; n++ ) {
		
		//-- test: head-on impact of two bird flocks
		/* bool ok = false;
		while (!ok) {
			pos = m_rnd.randV3( -50, 50 );
			if (pos.Length() < 50 ) {				
				grp = (n % 2);
				pos += Vec3F( 0, 100, grp ? -70 : 70 );
				vel = Vec3F(  0,   0, grp ?  18 :-18 );
				h = grp ? 90 : -90;
				b = AddBird ( pos, vel, Vec3F(0, 0, h), 3); 
				//rb->clr = (grp==0) ? Vec4F(1,0,0,1) : Vec4F(0,1,0,1);
				ok = true;
			}
		} */ 
		
		// randomly distribute birds
		pos = m_rnd.randV3( -50, 50 );
		pos.y = pos.y * .5f + 50;
		
		vel = m_rnd.randV3( -20, 20 );		
		vel *= 7.5 / vel.Length (); 
		h = m_rnd.randF(-180, 180); 
		b = AddBird ( pos, vel, Vec3F(0, 0, h), 1 );  
		b->clr = Vec4F( (pos.x+100)/200.0f, pos.y/200.f, (pos.z+100)/200.f, 1.f ); 


	}

	// **** add predators

	for (int n = 0; n < numPoints_pred; n++) {
		// randomly distribute predators
		pos = m_rnd.randV3(-50, 50);
		pos.y = pos.y * .5f + 50;
		vel = m_rnd.randV3(-20, 20);
		h = m_rnd.randF(-180, 180);

		//p = AddPredator(Vec3F(n+1, 0, 0), Vec3F(0, 0, 0), Vec3F(0, 0, h), 3);				// add static predator; (0,0,0) for pos does not work
		p = AddPredator(pos, vel, Vec3F(0, 0, h), 3);
		// printf("Predator is added at: %f, %f, %f \n", p->pos.x, p->pos.y, p->pos.z);
		p->clr = Vec4F(0.804, 0.961, 0.008, 1);

	}
	
	// Initialize accel grid
	//
	m_Accel.bound_min = Vec3F(-100,   0, -100);
	m_Accel.bound_max = Vec3F( 100, 150,  100);

	//m_Accel.bound_min = Vec3F(-50,   0, -50);
	//m_Accel.bound_max = Vec3F( 50, 100,  50);
	m_Accel.psmoothradius = 12;
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
				cuCheck ( cuModuleGetGlobal ( &m_cuFlock, &len, m_Module, "FFlock" ), "Initialize", "cuModuleGetGlobal", "cuFlock", true );
			}
			// Assign GPU symbols
			m_Birds.AssignToGPU ( "FBirds", m_Module );
			m_BirdsTmp.AssignToGPU ( "FBirdsTmp", m_Module );
			m_Grid.AssignToGPU ( "FGrid", m_Module );		
			m_Predators.AssignToGPU ( "FPredators", m_Module );			// predators
			cuCheck ( cuMemcpyHtoD ( m_cuAccel, &m_Accel,	sizeof(Accel) ),	"Accel", "cuMemcpyHtoD", "cuAccel", DEBUG_CUDA );
			cuCheck ( cuMemcpyHtoD ( m_cuParam, &m_Params, sizeof(Params) ),"Params", "cuMemcpyHtoD", "cuParam", DEBUG_CUDA );
			cuCheck ( cuMemcpyHtoD ( m_cuFlock, &m_Flock, sizeof(Flock) ),	"Flock", "cuMemcpyHtoD", "cuFlock", DEBUG_CUDA );

			// Commit birds
			m_Birds.CommitAll ();
			m_Predators.CommitAll();					// predators

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
			m_Predators.UpdateGPUAccess();		// predators

		}
	#endif

	printf ("Added %d birds.\n", m_Params.num_birds );
	printf ("Added %d predators.\n", m_Params.num_predators);		// predators

	// reset time
	m_time = 0;
	m_frame = 0;
	
	// clear plots
	m_vis.clear ();
	m_graph.clear ();
	m_plot[0].Fill ( 0,0,0,0 );	
	m_plot[1].Fill ( 0,0,0,0 );	

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
	int numPoints_pred = m_Params.num_predators; 		// ******

	int mem_usage = (m_gpu) ? DT_CPU | DT_CUMEM : DT_CPU;

	// Allocate acceleration
	m_Grid.DeleteAllBuffers ();
	m_Grid.AddBuffer ( AGRID,			"grid", sizeof(uint), numPoints, mem_usage );
	m_Grid.AddBuffer ( AGRIDCNT,	"gridcnt",	sizeof(uint), m_Accel.gridTotal, mem_usage );	
	m_Grid.AddBuffer ( AGRIDOFF,	"gridoff",	sizeof(uint), m_Accel.gridTotal, mem_usage );
	m_Grid.AddBuffer ( AAUXARRAY1,"aux1", sizeof(uint), numElem2, mem_usage );
	m_Grid.AddBuffer ( AAUXSCAN1, "scan1", sizeof(uint), numElem2, mem_usage );
	m_Grid.AddBuffer ( AAUXARRAY2,"aux2", sizeof(uint), numElem3, mem_usage );
	m_Grid.AddBuffer ( AAUXSCAN2, "scan2", sizeof(uint), numElem3, mem_usage );		

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
	int numPoints_pred = m_Params.num_predators; 		// *****


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
		memset( m_Birds.bufUI(FGNDX),	0,	numPoints*sizeof(int));		

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
		uint* pgcell = m_Birds.bufUI (FGCELL);
		uint* pgndx = m_Birds.bufUI (FGNDX);		
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
	
		uint* grid = m_Grid.bufUI(AGRID);
		uint* gridcnt = m_Grid.bufUI(AGRIDCNT);
		uint* fgc = m_Grid.bufUI(FGCELL);

		Bird *bi, *bj;

		float ang, birdang;		
		
		// topological distance
		float sort_d_nbr[12];
		int sort_j_nbr[12];
		int sort_num = 0;
		sort_d_nbr[0] = 10^5;
		sort_j_nbr[0] = -1;		
		int k, m;

		// for each bird..
		int numPoints = m_Params.num_birds;
		for (int i=0; i < numPoints; i++) {

			bi = (Bird*) m_Birds.GetElem( FBIRD, i);
			posi = bi->pos;

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
							if (j >= numPoints) continue;
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
		
	}
}
//----------------------------------------------------------------
void Sample::TrackBird() {

	float d = m_Accel.sim_scale;
	float d2 = d * d;
	float rd2 = (m_Accel.psmoothradius * m_Accel.psmoothradius) / d2;
	int	nadj = (m_Accel.gridRes.z + 1) * m_Accel.gridRes.x + 1;
	//uint j, cell;
	Vec3F bpos, ppos, dist;
	Vec3F diri, dirj;
	Vec3F cdir;
	//float dsq;
	//float nearest, nearest_fwd;
	float disti;

	uint* grid = m_Grid.bufUI(AGRID);
	uint* gridcnt = m_Grid.bufUI(AGRIDCNT);
	uint* fgc = m_Grid.bufUI(FGCELL);

	Bird* b;
	Predator* p;
	float closest;
	float predang;
	//float birdang;

	int numPoints = m_Params.num_birds;
	int numPoints_pred = m_Params.num_predators;

	for (int i = 0; i < numPoints_pred; i++) {
		p = (Predator*)m_Predators.GetElem(FPREDATOR, i);
		ppos = p->pos;
		closest = 1000;
		for (int j = 0; j < numPoints; j++) {
			b = (Bird*)m_Birds.GetElem(FBIRD, j);
			bpos = b->pos;
			// printf("pposj = %f, %f, %f ; ", pposj.x, pposj.y, pposj.z);

			dist = bpos - ppos;
			disti = dist.Length();

			dirj = dist.Normalize();
			diri = p->vel; diri.Normalize();
			predang = diri.Dot(dirj);

			if (disti < closest && predang > m_Params.fovcos_pred) {		// added predang, similar to findneighbors.birdang
				// printf("dist = %f; bird: %i; predang: %f\n\n", disti, j, predang);
				bird_index = j;
				closest = disti;
				closest_bird = closest;

			}
		}
		// b = (Bird*)m_Birds.GetElem(FBIRD,bird_index);
		// b->clr = Vec4F(0,0,1,1);
	}
}

void Sample::transitionPredState(int centroidReached, predState& currentState) {
	// printf("Entered transition.\n");
	if (centroidReached == 1) {
		currentState = HOVER;
	}
	else if (centroidReached == 2) {
		currentState = ATTACK;
	}
	else if (centroidReached == 3) {
		currentState = FOLLOW;
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

void Sample::UpdateFlockData ()
{	
	Vec3F centroid (0,0,0);	
	float speed = 0;	
	float plift = 0, pdrag = 0;
	float pfwd = 0, pturn=0, ptotal = 0;

	// compute centroid & energy of birds
	Bird* b;	
	for (int i=0; i < m_Params.num_birds; i++) {
		b = (Bird*) m_Birds.GetElem( FBIRD, i);		
		int gc = m_Birds.bufUI(FGCELL)[i];
		if ( gc != GRID_UNDEF ) {		
			centroid += b->pos;
			speed += b->speed;
			plift += b->Plift;
			pdrag += b->Pdrag;
			pfwd  += b->Pfwd;
			pturn += b->Pturn;
			ptotal += b->Ptotal;
		}
	}
	centroid *= (1.0f / m_Params.num_birds);	
		
	m_Flock.centroid = centroid;
	m_Flock.speed = speed / m_Params.num_birds;
	m_Flock.Plift = plift / m_Params.num_birds;
	m_Flock.Pdrag = pdrag / m_Params.num_birds;	
	m_Flock.Pfwd =  pfwd / m_Params.num_birds;	
	m_Flock.Pturn = pturn / m_Params.num_birds;	
	m_Flock.Ptotal = ptotal / m_Params.num_birds;		

	if ( m_frame > m_start_frame ) {
		if ( m_frame % 8 == 0 ) {
			float xscal = 1.0 / (m_Params.DT * 8.0f);
			float yscal = (m_method==0) ? 4e-4 : 5e-2;
			// Graph ( 0, m_Flock.Pturn, Vec4F(0,0,0,1), Vec2F(xscal, yscal) );
		}
	}

	if ( m_gpu ) {
		#ifdef BUILD_CUDA
			// transfer flock data to GPU, eg. centroid
			cuCheck ( cuMemcpyHtoD ( m_cuFlock, &m_Flock, sizeof(Flock) ),	"Flock", "cuMemcpyHtoD", "cuFlock", DEBUG_CUDA );

			// transfer predators to GPU
			m_Predators.CommitAll ();
		#endif
	}
}

typedef std::vector< Vec2F >	points_t;

bool fit( double& A, double& B, double& C, points_t const& pnts )
{
    if( pnts.size() < 2 ) { return false; }

    double X=0, Y=0, XY=0, X2=0, Y2=0;

		// Do all calculation symmetric regarding X and Y
    for (int n=0; n < pnts.size(); n++) {
        X  += pnts[n].x;
        Y  += pnts[n].y;
        XY += pnts[n].x * pnts[n].y;
        X2 += pnts[n].x * pnts[n].x;
        Y2 += pnts[n].y * pnts[n].y;
    }
    X  /= pnts.size();
    Y  /= pnts.size();
    XY /= pnts.size();
    X2 /= pnts.size();
    Y2 /= pnts.size();
    A = - ( XY - X * Y );			// Common for both solution
    double Bx = X2 - X * X;
    double By = Y2 - Y * Y;

    if( fabs( Bx ) < fabs( By ) )	{	// Line is more Vertical.    
        B = By;
        std::swap(A,B);
    } else {												// Line is more Horizontal.																		
        B = Bx;											// Classical solution, when we expect more horizontal-like line
    }
    C = - ( A * X + B * Y );

    // Optional normalization:
    double D = sqrt( A*A + B*B );
    A /= D;
    B /= D;
    C /= D;
    return true;
}

void Sample::StartNextRun ()
{
	// record the last run
	// printf ( "run, num_run, val, #bird, #peaks, peak_ave, g0_min,g0_max, g1_min,g1_max, g2_min,g2_max, g3_min,g3_max\n" );
	if (m_run >= 0) {
	  fprintf ( m_runs_outfile, "%d,%d,%f, %d,%d,%f, %f, %f,%f, %f,%f, %f,%f, %f,%f\n", m_run, m_num_run, m_val.z, m_Params.num_birds, m_peak_cnt, m_peak_ave, m_peak_max, 
		  m_freq_gmin[0],m_freq_gmax[0], m_freq_gmin[1],m_freq_gmax[1], m_freq_gmin[2],m_freq_gmax[2], m_freq_gmin[3],m_freq_gmax[3] );

		// close & reopen to save 
		fclose ( m_runs_outfile );
		m_runs_outfile = fopen ( "output.csv", "a" );	
	}	
	
	// advance run
	m_run++;

	// replace this line with the parameter you wish to test
	m_val.z = m_val.x + (m_val.y-m_val.x) * float(m_run) / m_num_run;

	m_Params.reynolds_alignment = m_val.z;
	
	//m_Params.align_amt = m_val.z;

	// reset simulation
	Reset ( m_Params.num_birds, m_Params.num_predators );

	printf ( "Run: %d/%d, #Bird: %d, Val: %f\n", m_run, m_num_run, m_Params.num_birds, m_val.z );
}

void Sample::OutputFFTW ( int frame )
{
	Bird* b;
	float ang_accel;
	Vec4F c;
	float fm, fr, fi, v;
	double fmag[ PLOT_RESY ];	
	float freq_wgt_ave;
	int xi, y;

	int N = m_fftw_N;	

	float scalar = (m_method==0) ? 1 : 100.;  

	// waiting for experiment to start
	xi = frame - m_start_frame;
	if ( xi < 0 || xi >= SAMPLES ) return;

	// experiment complete
	if ( frame > m_end_frame ) {
		StartNextRun ();
		return;
	}

	// running experiment (measuring)
	m_draw_plot = true;

	// initialize freq accumulator
	for (int f=0; f < N; f++) {
		fmag[f] = 0;
	}
	
	// build sample matrix
	// x-axis = time
	// y-axis = bird id
	// f(x,y) = bird angular accel at time x
	double* s = m_samples + xi;									// for a given time x (column)
	float ave = 0;
	for (int i=0; i < m_Params.num_birds; i++) {
		b = (Bird*) m_Birds.GetElem( FBIRD, i );		
		y = b->id;		
		ang_accel = b->ang_accel.Length();				// sample from angular acceleration
		if ( y > 0 && y < MAX_BIRDS) {
			*(s + y*SAMPLES) = ang_accel * scalar;
		}
		ave += ang_accel;
	}
	ave /= m_Params.num_birds;


	// STFT using windowed FFT	
	if ( xi > N ) {
		for (y=0; y < m_Params.num_birds; y++) {					
			// capture real-valued window			
			s = m_samples + y*SAMPLES;
			for (int k=0; k < N; k++) {
				m_fftw_in[k] = s[ xi-N+k ] * 0.5 * (1- cos(2*PI*double(k)/(N-1)));		// moving window, with hanning filter
			}
			// execute fftw			
			fftw_execute ( m_fftw_plan );

			// accumulate freq magnitudes			
			for (int f=0; f < N/2; f++) {
				fr = m_fftw_out[f][0] * 2.0/N;			// real part
				fi = m_fftw_out[f][1] * 2.0/N;			// imaginary part
				fm = fr*fr + fi*fi;									// magnitude of given freq
				fmag[f] += fm;				
			}
		}	
	}

	// x-coordinate
	int x, xf;
	int xdiv = 8;
	x = xi / xdiv;	
	xf = (xi-N/2)/xdiv;
	
	if ( xi >= N/2 ) {
		float energy = 0.0;		
		float emin = 0;
		
		for (int g=0; g<4; g++) {
			m_freq_grp[xi][g] = 0;
		}
		
		// spectrogram - plot power of frequencies for current time		
		for (int f=1; f < N/2; f++) {					
			v = fmax(0, fmin( 1, 0.01f * 10. * log(fmag[f] + 1e-6) / log(10) ));		// dB						
			v = v*v / 5.f;
			if (f < N/4) {
				energy += v;				
			}
			for (int g=0; g < 4; g++) {
				if ( (g/4.0)*(N/2) < f && f < ((g+1)/4.0)*(N/2) ) {
					m_freq_grp[xi][g] += v;
				}
			}
			c = m_plot[0].GetPixel ( xf, f );
			c += Vec4F(v,v,v,1);
			m_plot[0].SetPixel ( xf, f, c );
		}				
		
		// plot and record total spectral energy
		c = Vec4F(0,0,0,1);			
		energy = energy * 0.5 / (N/256.0f); // - 1.2f;
		m_fftw_energy [ xf ] = energy;					

		// plot weighted ave. frequency 
		Vec4F clrgrp[4];
		clrgrp[0] = Vec4F(1,0,0,1);
		clrgrp[1] = Vec4F(1,1,0,1);
		clrgrp[2] = Vec4F(0,1,0,1);
		clrgrp[3] = Vec4F(0,0,1,1);
		for (int g=0; g < 4; g++) {
			m_freq_grp[xi][g] = m_freq_grp[xi][g] * 0.5 / (N/256.0f);
			m_plot[0].SetPixel ( xf, PLOT_RESY - m_freq_grp[xi][g]*400, clrgrp[g] );		
		}		
		//printf ( "%f, %f\n", m_fftw_energy[xf], energy );
		

		// 200 hz = 1 sec		
		if ( xi % 200 == 0 ) {
			
			// run analysis
			char txt[512];
			int js = 0; //N/2)/xdiv;			// xf-25;
			float* e;
			float diff;

			// clear analysis
			points_t pnts;
			m_vis.clear();						// clear peak markers
			m_lines.clear ();					// clear fit line(s)
			
			// pad first N/2 samples
			for (int j = 0; j <= N/2; j++) {
				m_fftw_energy[j] = m_fftw_energy[N/2];
			}
			for (int j = 0; j <= N; j++) {
				m_freq_grp[j][0] = m_freq_grp[N+1][0];
				m_freq_grp[j][1] = m_freq_grp[N+1][1];
				m_freq_grp[j][2] = m_freq_grp[N+1][2];
				m_freq_grp[j][3] = m_freq_grp[N+1][3];
			}
			// smooth the energy func
			memcpy (m_fftw_s1, m_fftw_energy, sizeof(float) * xf );			
			for (int iter=0; iter < 20; iter++) {
				m_fftw_s1[0] = m_fftw_s1[1];
				m_fftw_s1[xf-1] = m_fftw_s1[xf-2];
				for (int j = 1; j <= xf-2; j++) {
					m_fftw_s2[j] = m_fftw_s1[j-1]*0.3 + m_fftw_s1[j]*0.4 + m_fftw_s1[j+1]*0.3;
				}
				memcpy ( m_fftw_s1, m_fftw_s2, sizeof(float) * xf );
			}
			// collect energy func as points
			for (int j = 0; j < xf; j++) {
				pnts.push_back ( Vec2F( j, m_fftw_s2[j] ) );				
				m_plot[0].SetPixel ( j, PLOT_RESY - m_fftw_s2[j]*400, c );		
			}
			// fit a line to energy
			double A, B, C, m, b;
			if ( fit( A, B, C, pnts ) ) {
				m = (-A/B);
				b = (-C/B);
				m_lines.push_back ( Vec4F(0, PLOT_RESY-b*400, xf, PLOT_RESY-(m*xf+b)*400 ) );
			}
			// count peaks in energy			
			e = m_fftw_s2;		
			m_peak_cnt = 0;
			m_peak_ave = 0;			
			m_peak_max = 0;
			for (int j = 3; j < xf; j++) {
				diff = fabs( e[3] - (m*j+b) );
				if ( e[0] < e[2] && e[2] < e[3] && e[3] > e[4] && e[4] > e[6] && diff > 0.01 ) { 					
					// compute diff to line					
					sprintf ( txt, "%4.1f", diff*100.0f);
					m_vis.push_back ( vis_t( Vec3F( j, PLOT_RESY-e[3]*400, 0), 2.0f, Vec4F(0,0,0,1), txt ) );
					if ( diff*100.0 > m_peak_max ) m_peak_max = diff*100.0f;
					m_peak_ave += diff*100.0f;
					m_peak_cnt++;					
				}
				e++;
			}
			if ( m_peak_cnt > 0 ) {
				m_peak_ave /= m_peak_cnt;				
			}
			printf ( "%d %f %f\n", m_peak_cnt, m_peak_ave, m_peak_max );
			
			// measure min/max frequency groups
			for (int g=0; g < 4; g++) {
				m_freq_gmin[g] = m_freq_grp[1][g];
				m_freq_gmax[g] = m_freq_grp[1][g];				
				for (int j = 1; j <= xi; j++) {
					if ( m_freq_grp[j][g] < m_freq_gmin[g] ) m_freq_gmin[g] = m_freq_grp[j][g];
					if ( m_freq_grp[j][g] > m_freq_gmax[g] ) m_freq_gmax[g] = m_freq_grp[j][g];
				}				
			}
			m_lines.push_back ( Vec4F(0, PLOT_RESY-m_freq_gmin[0]*400, xf, PLOT_RESY-m_freq_gmin[0]*400 ) );
			m_lines.push_back ( Vec4F(0, PLOT_RESY-m_freq_gmax[0]*400, xf, PLOT_RESY-m_freq_gmax[0]*400 ) );
		}
	}

	// plot samples
	s = m_samples + (N/2)*SAMPLES + x;
	for (y = N/2; y < N; y++) {			
		v = (*s) * 0.05f / 5.f;
		c = m_plot[0].GetPixel ( x, y);		
		c += Vec4F(v, v, v, 1); 
		m_plot[0].SetPixel ( x, y, c );
		s += SAMPLES;
	}	

	if ( xi % xdiv == 0 ) {
		m_plot[0].Commit ();
	}
}


// compute smoothed energy
/* memcpy ( m_fftw_s1, m_fftw_energy, xf * sizeof(float) );
for (int iter=0; iter < 500; iter++) {
	for (int j=2; j < xf-1; j++) {
		m_fftw_s2[ j ] = m_fftw_s1[j-1]*0.3 + m_fftw_s1[j]*0.4 + m_fftw_s1[j+1]*0.3;
	}
	memcpy ( m_fftw_s1, m_fftw_s2, xf*sizeof(float) );
}
c = Vec4F(1,1,1,1);
for (int j=1; j < xf-1; j++) {
	m_plot[0].SetPixel ( j, PLOT_RESY - m_fftw_s2[j], c );		
} */

void Sample::OutputPlot ( int what, int frame )
{	
	Bird* b;
	float ang_accel;
	Vec4F c;
	int x, y;

	x = frame / 5;
	if ( x >= PLOT_RESX ) return;
	
	for (int i=0; i < m_Params.num_birds; i++) {
		b = (Bird*) m_Birds.GetElem( FBIRD, i );
		y = min( b->id, PLOT_RESY );

		if ( y < PLOT_RESY ) {
			ang_accel = b->ang_accel.Length() * .002;		// 60 - classic			
			c = m_plot[0].GetPixel ( x, y );
			c += Vec4F(ang_accel, 0, 0, 0);
			m_plot[0].SetPixel (x, y, c ); 
		}
	}
	m_plot[0].Commit ();
}

void Sample::OutputPointCloudFiles ( int frame )
{
	Bird* b;	
	FILE* fp;
	char fn[512];

	// write flock data as PLY point cloud	
	//
	// how to read and plot with MATLAB:
	//   frame = 1
	//   pts = pcread ( "birds"+num2str(frame,'%04d')+".ply")
	//   pcshow ( pts.Location, pts.Normal)  
	//

	// only record certain frames..
	// - m_rec_start, skip until flocking settles
	// - m_rec_step, skip every n-th frame
	if ( frame > m_rec_start && (frame % m_rec_step)==0 ) {
		
		// make file numbers continuous
		int file_num = (frame - m_rec_start)/m_rec_step;

		sprintf ( fn, "birds%04d.ply", file_num );
		fp = fopen ( fn, "wt" );
		if (fp==0) return;
		fprintf ( fp, "ply\n" );
		fprintf ( fp, "format ascii 1.0\n" );
		fprintf ( fp, "element vertex %d\n", m_Params.num_birds );
		fprintf ( fp, "property float x\n" );
		fprintf ( fp, "property float y\n" );
		fprintf ( fp, "property float z\n" );		
		fprintf ( fp, "property float nx\n" );
		fprintf ( fp, "property float ny\n" );
		fprintf ( fp, "property float nz\n" );		
		fprintf ( fp, "end_header\n" );
		// xyz (position) is the bird position
		// nx,ny,nz (normal) is saved as the bird angular acceleration (but could store other bird variables)
		// note: Y+ is up in simulation, exported with Z+ up
		for (int i=0; i < m_Params.num_birds; i++) {
			b = (Bird*) m_Birds.GetElem( FBIRD, i);				
			fprintf ( fp, "%4.3f %4.3f %4.3f %4.3f %4.3f %4.3f\n", b->pos.x, b->pos.z, b->pos.y, b->ang_accel.x, b->ang_accel.z, b->ang_accel.y );
		}
		fclose ( fp );
	}

}



void Sample::AdvanceByOrientation ()
{
	if (m_gpu) {

		#ifdef BUILD_CUDA
			// Advance - GPU
			//			
			void* args[4] = { &m_time, &m_Params.DT, &m_Accel.sim_scale, &m_Params.num_birds };

			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_ADVANCE_ORIENT],  m_Accel.numBlocks, 1, 1, m_Accel.numThreads, 1, 1, 0, NULL, args, NULL), "Advance", "cuLaunch", "FUNC_ADVANCE", DEBUG_CUDA );

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
		Predator* p;
		bool leader;
		float ave_yaw;

		Vec3F centroid (0,50,0);
		
		for (int n=0; n < m_Params.num_birds; n++) {

			b = (Bird*) m_Birds.GetElem( FBIRD, n);

			b->clr.Set(0,0,0,0);	

			// Hoetzlein - Peripheral bird term
			// Turn isolated birds toward flock centroid
			float d = b->r_nbrs / m_Params.boundary_cnt;
			if ( d < 1 ) {
				b->clr.Set(1,.5,0, 1);	
				dirj = centroid - b->pos; dirj.Normalize();
				dirj *= b->orient.inverse();
				yaw = atan2( dirj.z, dirj.x )*RADtoDEG;
				pitch = asin( dirj.y )*RADtoDEG;
				b->target.z +=   yaw * m_Params.boundary_amt;
				b->target.y += pitch * m_Params.boundary_amt;
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
			
			// RULE 4: bird-predator behaviour
			///*
			for (int m = 0; m < m_Params.num_predators; m++) {
				p = (Predator*)m_Predators.GetElem(FPREDATOR, m);
				Vec3F predatorDir = p->pos - b->pos;
				float predatorDist = predatorDir.Length();

				if (predatorDist < m_Params.pred_radius) {
					// Flee from predator
					//predatorDir.Normalize();

					predatorDir = (predatorDir / predatorDist) * b->orient.inverse();
					yaw = atan2(predatorDir.z, predatorDir.x) * RADtoDEG;
					pitch = asin(predatorDir.y) * RADtoDEG;
					predatorDist = fmax(1.0f, fmin(predatorDist * predatorDist, 100.0f));
					b->target.z -= yaw * m_Params.avoid_pred_angular_amt; // / predatorDist;
					b->target.y -= pitch * m_Params.avoid_pred_angular_amt; // / predatorDist;
					b->clr = Vec4F(1, 0, 1, 1);
					bird_count += 1;
				}
				
			}
		}

		//--- Flight model
		//
		for (int n=0; n < m_Params.num_birds; n++) {

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
			if ( b->speed < m_Params.min_speed ) {
				b->speed = m_Params.min_speed;				// birds dont go in reverse
			}
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
			float rx = m_Params.DT*1000.0f / m_Params.reaction_speed;
			ctrlq.fromAngleAxis ( b->ang_accel.x * rx, fwd );
			b->orient *= ctrlq;	b->orient.normalize();

			// Pitch & Yaw - Control inputs
			// - apply 'torque' by rotating the velocity vector based on pitch & yaw inputs				
			ctrlq.fromAngleAxis ( b->ang_accel.z * rx, up * -1.f );
			vaxis *= ctrlq; vaxis.Normalize();	
			ctrlq.fromAngleAxis ( b->ang_accel.y * rx, right );
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
			L = (sin( aoa * 0.1)+0.5) * dynamic_pressure * m_Params.lift_factor *m_Params.wing_area;		// lift equation. L = CL (1/2 p v^2) A			
			lift = up * L;
			force += lift;

			// Drag force	
			drag = vaxis * dynamic_pressure * -m_Params.drag_factor  * m_Params.wing_area;			// drag equation. D = Cd (1/2 p v^2) A
			force += drag; 

			// Thrust force
			thrust = fwd * b->power * m_Params.power;
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

void Sample::AdvanceByVectors ()
{
	if (m_gpu) {

		#ifdef BUILD_CUDA
			// Advance - GPU
			//			
			void* args[4] = { &m_time, &m_Params.DT, &m_Accel.sim_scale, &m_Params.num_birds };
			cuCheck ( cuLaunchKernel ( m_Kernel[KERNEL_ADVANCE_VECTORS],  m_Accel.numBlocks, 1, 1, m_Accel.numThreads, 1, 1, 0, NULL, args, NULL), "Advance", "cuLaunch", "FUNC_ADVANCE", DEBUG_CUDA );

			// Retrieve birds from GPU for rendering & visualization
			m_Birds.Retrieve ( FBIRD );
		
			cuCtxSynchronize ();
		#endif

	} else {

		// Advance - CPU
		// Using classic Reynold's vector-based boids model 
		//  1987, Craig Reynolds. "Flocks, herds and schools: A distributed behavioral model"
		//
		Vec3F dirj, force, accel;
		Bird *b, *bj, *bfwd;
		Predator* p;
	
		//--- Reynold's model
		//
		for (int n=0; n < m_Params.num_birds; n++) {

			b = (Bird*) m_Birds.GetElem( FBIRD, n);

			force.Set(0,0,0);
			
			// Rule #1 - Avoidance			
			// avoid nearest bird
			if ( b->near_j != -1) {
					// get nearest bird
					bj = (Bird*) m_Birds.GetElem(0, b->near_j);
					dirj = bj->pos - b->pos;
					dirj.Normalize ();
					force -= dirj * m_Params.reynolds_avoidance;
			}
			// Rule #2. Alignment 
			dirj = b->ave_vel - b->vel;
			force += dirj * m_Params.reynolds_alignment;

			// Rule #3. Cohesion
			dirj = b->ave_pos - b->pos;
			force += dirj * m_Params.reynolds_cohesion;		

			// Integrate position	& velocity 
			accel = force / m_Params.mass;	
			b->vel += accel * m_Params.DT;	
			b->pos += b->vel * m_Params.DT;

			// Boundaries
			if ( b->pos.x < m_Accel.bound_min.x ) b->pos.x = m_Accel.bound_max.x;
			if ( b->pos.x > m_Accel.bound_max.x ) b->pos.x = m_Accel.bound_min.x;
			if ( b->pos.z < m_Accel.bound_min.z ) b->pos.z = m_Accel.bound_max.z;
			if ( b->pos.z > m_Accel.bound_max.z ) b->pos.z = m_Accel.bound_min.z;

		}
	}

}


// -----------------------PREDATOR------------------------- 

void Sample::Advance_pred() 
{
	// Advance - CPU
	//
	Vec3F   fwd, up, right, vaxis;
	Vec3F   force, lift, drag, thrust, accel;
	Vec3F   diri, dirj, unNormDirj, dirf;
	Quaternion ctrl_pitch;
	float   airflow, dynamic_pressure, aoa;
	float   L, dist;
	float   pitch, yaw;
	Quaternion ctrlq, tq;
	Vec3F   angs;
	Quaternion angvel;
	//Bird *b, *bj, *bfwd;
	Predator* p, * pj;
	float   dist_target_bird;
	predState new_state;

	yaw = 0;
	pitch = 0;

	m_predcentroid.Set(0, 25, 25);

	for (int n = 0; n < m_Params.num_predators; n++) {

		p = (Predator*) m_Predators.GetElem(FPREDATOR, n);

		dirj = m_Flock.centroid - p->pos;		
		dist = dirj.Length();
		dirj.Normalize();
		dirj *= p->orient.inverse();		

		new_state = p->currentState;		// assume same state for now


		if ( dist > 0) {
			// only move predator if dist > 0
			// (otherwise causes 'nan', div by 0)
			
			if (p->currentState == HOVER) {
				//printf("current state = HOVER\n");

				// dirj = (dirj / dist) * p->orient.inverse();
				yaw = atan2(dirj.z, dirj.x) * RADtoDEG;
				pitch = asin(dirj.y) * RADtoDEG;
				p->target.z -= yaw * m_Params.avoid_pred_angular_amt;
				p->target.y -= pitch * m_Params.avoid_pred_angular_amt;
			
				if (dist > 10.0f) {			
					new_state = ATTACK;				// predator far from flock, switch to attack
					//printf("Distance reached, %f.\n", p->pos.y);
				}

			}
			else if (p->currentState == ATTACK) {
				//printf("current state = ATTACK\n");
				yaw = atan2(dirj.z, dirj.x) * RADtoDEG;
				pitch = asin(dirj.y) * RADtoDEG;
				p->target.z += yaw * m_Params.boundary_amt;
				p->target.y += pitch * m_Params.boundary_amt;

				if (dist < 1.0f) {
					new_state = HOVER;			// predator close to centroid, switch to hover
					//printf("Centroid reached.\n");
				}

			}
			else if (p->currentState == FOLLOW) { // 3 : Follow

				//printf("current state = FOLLOW\n");
				Bird* b = (Bird*)m_Birds.GetElem(FBIRD, bird_index);
				dirf = b->pos - p->pos;
				dist_target_bird = dirf.Length();
				dirf.Normalize();
				dirf *= p->orient.inverse();

				yaw = atan2(dirj.z, dirj.x) * RADtoDEG;
				pitch = asin(dirj.y) * RADtoDEG;
				p->target.z += yaw * m_Params.boundary_amt;
				p->target.y += pitch * m_Params.boundary_amt;

				//printf("Following bird\n");

				if (dist_target_bird < 5.5f) {				
					new_state = HOVER;			// target bird caught, switch to eatting!
				} else if (dist < 5.5f) { 
					new_state = HOVER;			// another bird caught, switch to hover
				}
			}
		}

		// update predator state
		p->currentState = new_state;

	}

	//--- Flight model
	//
	for (int n = 0; n < m_Params.num_predators; n++) {

		p = (Predator*)m_Predators.GetElem(FPREDATOR, n);

		// Body orientation
		fwd = Vec3F(1, 0, 0) * p->orient;			// X-axis is body forward
		up = Vec3F(0, 1, 0) * p->orient;			// Y-axis is body up
		right = Vec3F(0, 0, 1) * p->orient;		// Z-axis is body right

		// Direction of motion
		p->speed = p->vel.Length();
		if (p->speed < m_Params.min_predspeed) p->speed = m_Params.min_predspeed;				// birds dont go in reverse // set min speed to predminspeed
		if (p->speed > m_Params.max_predspeed) p->speed = m_Params.max_predspeed;
		if (p->speed == 0) {
			vaxis = fwd;		
		} else {
			vaxis = p->vel / p->speed;
		}
		if ( isnan(vaxis.x) ) {
			bool stop=true;
		}		

		p->orient.toEuler(angs);

		// Target corrections
		angs.z = fmod(angs.z, 180.0);
		p->target.z = fmod(p->target.z, 180);										// yaw -180/180
		p->target.x = circleDelta(p->target.z, angs.z) * 0.5;				// banking
		p->target.y *= m_Params.pitch_decay;																				// level out
		if (p->target.y < m_Params.pitch_min) p->target.y = m_Params.pitch_min;
		if (p->target.y > m_Params.pitch_max) p->target.y = m_Params.pitch_max;
		if (fabs(p->target.y) < 0.0001) p->target.y = 0;

		// Compute angular acceleration
		// - as difference between current direction and desired direction
		p->ang_accel.x = (p->target.x - angs.x);
		p->ang_accel.y = (p->target.y - angs.y);
		p->ang_accel.z = circleDelta(p->target.z, angs.z);

		// Roll - Control input
		// - orient the body by roll
		ctrlq.fromAngleAxis(p->ang_accel.x * m_Params.reaction_speed, fwd);
		p->orient *= ctrlq;	p->orient.normalize();

		// Pitch & Yaw - Control inputs
		// - apply 'torque' by rotating the velocity vector based on pitch & yaw inputs				
		ctrlq.fromAngleAxis(p->ang_accel.z * m_Params.reaction_speed, up * -1.f);
		vaxis *= ctrlq; vaxis.Normalize();
		ctrlq.fromAngleAxis(p->ang_accel.y * m_Params.reaction_speed, right);
		vaxis *= ctrlq; vaxis.Normalize();	

		// Adjust velocity vector		
		p->vel = vaxis * p->speed;

		force = 0;

		// Dynamic pressure		
		airflow = p->speed + m_Params.wind.Dot(fwd * -1.0f);		// airflow = air over wing due to speed + external wind			
		dynamic_pressure = 0.5f * m_Params.air_density * airflow * airflow;

		// Lift force
		aoa = acos(fwd.Dot(vaxis)) * RADtoDEG + 1;		// angle-of-attack = angle between velocity and body forward		
		if (isnan(aoa)) aoa = 1;
		// CL = sin(aoa * 0.2) = coeff of lift, approximate CL curve with sin
		L = sin(aoa * 0.2) * dynamic_pressure * m_Params.lift_factor * 0.5;		// lift equation. L = CL (1/2 p v^2) A
		lift = up * L;
		force += lift;

		// Drag force	
		drag = vaxis * dynamic_pressure * m_Params.drag_factor * -1.0f;			// drag equation. D = Cd (1/2 p v^2) A
		force += drag;

		// Thrust force
		thrust = fwd * p->power;
		force += thrust;

		// Integrate position		
		accel = force / m_Params.pred_mass;				// body forces	// changed mass from 0.1 to 0.8
		accel += m_Params.gravity;						// gravity
		accel += m_Params.wind * m_Params.air_density * m_Params.front_area;		// wind force. Fw = w^2 p * A, where w=wind speed, p=air density, A=frontal area

		p->pos += p->vel * m_Params.DT;

		// Boundaries
		if (p->pos.x < m_Accel.bound_min.x) p->pos.x = m_Accel.bound_max.x;
		if (p->pos.x > m_Accel.bound_max.x) p->pos.x = m_Accel.bound_min.x;
		if (p->pos.z < m_Accel.bound_min.z) p->pos.z = m_Accel.bound_max.z;
		if (p->pos.z > m_Accel.bound_max.z) p->pos.z = m_Accel.bound_min.z;

		// Ground avoidance
		L = p->pos.y - m_Accel.bound_min.y;
		if (L < m_Params.bound_soften) {
			L = (m_Params.bound_soften - L) / m_Params.bound_soften;
			p->target.y += L * m_Params.avoid_ground_amt;
			// power up so we have enough lift to avoid the ground
			p->power = m_Params.avoid_ground_power;
		}

		// Ceiling avoidance
		L = m_Accel.bound_max.y - p->pos.y;
		if (L < m_Params.bound_soften) {
			L = (m_Params.bound_soften - L) / m_Params.bound_soften;
			p->target.y -= L * m_Params.avoid_ceil_amt;
		}

		// Ground condition
		if (p->pos.y <= 0.00001) {
			// Ground forces
			p->pos.y = 0; p->vel.y = 0;
			p->accel += Vec3F(0, 9.8, 0);	// ground force (upward)
			p->vel *= 0.9999;				// ground friction
			p->orient.fromDirectionAndRoll(Vec3F(fwd.x, 0, fwd.z), 0);	// zero pitch & roll			
		}

		// Integrate velocity
		p->vel += accel * m_Params.DT;

		vaxis = p->vel;	
		vaxis.Normalize();

		// Update Orientation
		// Directional stability: airplane will typically reorient toward the velocity vector
		//  see: https://github.com/ramakarl/Flightsim
		// this is an assumption yet much simpler/faster than integrating body orientation
		// this way we dont need torque, angular vel, or rotational inertia.
		// stalls are possible but not flat spins or 3D flying		
		angvel.fromRotationFromTo(fwd, vaxis, m_Params.dynamic_stability);
		if (!isnan(angvel.X)) {
			p->orient *= angvel;
			p->orient.normalize();
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

void Sample::Graph ( int id, float y, Vec4F clr, Vec2F scal)
{
	if ( id >= m_graph.size() ) {
		while (id >= m_graph.size()) {			
			graph_t ng;
			ng.x = 0;
			memset ( &ng.y[0], 0, 2048 * sizeof(float) );
			ng.clr = clr;
			ng.scal = scal;			
			m_graph.push_back ( ng );
		}
	}
	graph_t* g = &m_graph[id];
	g->x++;
	if ( g->x >= 2048 ) g->x = 0;
	g->y[ int(g->x) ] = y;
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

	if (ndx == -1 ) {
		dbgprintf ( "bird not found: %d\n", m_bird_sel);
		return;
	}

	m_bird_ndx = ndx;

	// bird information
	char msg[1024]; 
	Vec4F tc (1,1,1,1);
	sprintf ( msg, "thrust:  %4.3f N", b->thrust.Length() );	drawText ( Vec2F(10, 30), msg, tc );
	sprintf ( msg, "drag:    %4.3f N", b->drag.Length() );		drawText ( Vec2F(10, 50), msg, tc );
	sprintf ( msg, "lift:    %4.3f N", b->lift.Length() );		drawText ( Vec2F(10, 70), msg, tc );
	float LD = b->lift.Length() / b->drag.Length();
	sprintf ( msg, "L/D:     %4.1f", LD );				  					drawText ( Vec2F(10, 90), msg, tc );
	sprintf ( msg, "gravity: %4.3f N", b->gravity.Length() );	drawText ( Vec2F(10, 110), msg, tc );
	sprintf ( msg, "Plift:   %4.3f watts", b->Plift );				drawText ( Vec2F(10, 130), msg, tc );
	sprintf ( msg, "Pdrag:   %4.3f watts", b->Pdrag );				drawText ( Vec2F(10, 150), msg, tc );
	sprintf ( msg, "Pfwd:    %4.6f watts", b->Pfwd );				drawText ( Vec2F(10, 170), msg, tc );
	sprintf ( msg, "Pturn:   %4.6f watts", b->Pturn );			drawText ( Vec2F(10, 190), msg, tc );
	// Pfwd = Pprof + Ppara   (Pfwd = profile + parasitic power, and Pdrag is already included)
	float P = b->Plift + b->Pfwd + b->Pturn;
	sprintf ( msg, "Ptotal:  %4.3f watts", P );								drawText ( Vec2F(10, 210), msg, tc );
	sprintf ( msg, "speed:   %4.3f m/s", b->speed );					drawText ( Vec2F(10, 230), msg, tc );
	sprintf ( msg, "power:   %4.3f joules", b->power * m_Params.power ); drawText ( Vec2F(10, 250), msg, tc );		
		
	sprintf ( msg, "ave. lift:  %4.3f watts / bird", m_Flock.Plift );		drawText ( Vec2F(10, 280), msg, tc );
	sprintf ( msg, "ave. drag:  %4.3f watts / bird", m_Flock.Pdrag );		drawText ( Vec2F(10, 300), msg, tc );
	sprintf ( msg, "ave. fwd:   %4.6f watts / bird", m_Flock.Pfwd );		drawText ( Vec2F(10, 320), msg, tc );
	sprintf ( msg, "ave. turn:  %4.6f watts / bird", m_Flock.Pturn );		drawText ( Vec2F(10, 340), msg, tc );
	sprintf ( msg, "ave. total: %4.3f watts / bird", m_Flock.Ptotal );	drawText ( Vec2F(10, 360), msg, tc );
	sprintf ( msg, "ave. speed: %4.6f m/s", m_Flock.speed );						drawText ( Vec2F(10, 380), msg, tc );

	// visualize bird (green)
	m_vis.push_back ( vis_t( b->pos, 1.1f, Vec4F(0,1,0,1), "" ) );

	// visualize neighborhood radius (yellow)
	m_vis.push_back ( vis_t( b->pos, m_Accel.psmoothradius, Vec4F(1,1,0,1), "" ) );

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
		float dsq, birdang, ave_dist = 0;
		Vec3F dist, diri;
		diri = b->vel; diri.Normalize();
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
					dist = bj->pos - b->pos;
					dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
					
					if ( dsq < rd2 ) {							
						dsq = sqrt(dsq);
						dist /= dsq;
						birdang = diri.Dot ( dist );
						if ( birdang > m_Params.fovcos ) {
							ave_dist += dsq;
							ncnt++;
							m_vis.push_back ( vis_t( bj->pos, 0.5f, Vec4F(1,1,0,1), "" ) );		// neighbor birds (yellow)
						} else {
							m_vis.push_back ( vis_t( bj->pos, 0.5f, Vec4F(1,0,0,1), "" ) );		// neighbor birds (yellow)
						}
					}
			}
		}
		if ( ncnt > 0) {
			ave_dist /= ncnt;
			// printf ( "ave dist: %f\n", ave_dist );
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

	bird_count = 0;

	//--- Insert birds into acceleration grid
	InsertIntoGrid ();

	//--- Prefix scan for accel grid
	PrefixSumGrid ();

	//--- Find neighbors
	FindNeighbors ();

	//--- Advance birds with behaviors & flight model
	if ( m_method==0 ) {
		AdvanceByOrientation ();	
	} else {
		AdvanceByVectors ();
	}

	//--- Advance predators
	Advance_pred();	

	//--- Update flock data (centroid, energy)
	UpdateFlockData ();

	//--- Outputs 
	// OutputPointCloudFiles ( m_frame );  
	// OutputPlot ( 0, m_frame );
	// OutputFFTW ( m_frame );

	#ifdef DEBUG_BIRD
		DebugBird ( 7, "Post-Advance" );
	#endif

	// computation timing
	t2.SetTimeNSec();
	float msec = t2.GetElapsedMSec( t1 );
	// printf ( "Run: %f msec/step, %2.2f%% real-time\n", msec, (m_Params.DT*1000.0)*100 / msec );

	// PERF_POP();

	m_time += m_Params.DT;
	m_frame++;
	
	runcount += 1;

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
	if (!m_cam_adjust) {
		m_cam->setDirection ( m_cam->getPos(), m_Flock.centroid, 0 );
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
	
	m_running = true;

	m_cockpit_view = false;
	m_draw_sphere = false;
	m_draw_grid = false;
	m_draw_vis = 0;
	m_cam_mode = 0;	

	m_method = 0;					// 0 = Flock2, 1 = Reynolds

	m_rnd.seed (12);

	m_rec_start = 1000;
	m_rec_step = 10;

	m_time = 0;
	m_frame = 0;

	// Build FFTW arrays 
	#ifdef USE_FFTW		
		m_fftw_N = 64;
		m_fftw_in = (double*) malloc ( sizeof(double) * m_fftw_N );
		m_fftw_out = (fftw_complex*) fftw_malloc ( sizeof(fftw_complex) * m_fftw_N);
		m_fftw_plan = fftw_plan_dft_r2c_1d ( m_fftw_N, m_fftw_in, m_fftw_out, FFTW_ESTIMATE );	
		
		// *NOTE* m_samples matrix could be large. SAMPLES=16384, MAX_BIRD=65535,
		// samples = 8 bytes * 16384 * 65535 = 8.5 GB
		m_samples = (double*) malloc ( sizeof(double) * SAMPLES * MAX_BIRDS );

		memset ( m_fftw_energy, 0, 32767*sizeof(float) );		
	#endif

	// enable GPU if cuda available
	#ifdef BUILD_CUDA
		m_gpu = true;
  #else
	  m_gpu = false;
	#endif

	// override. use CPU 
	// m_gpu = false;

	m_plot[0].Resize ( PLOT_RESX, PLOT_RESY, ImageOp::RGBA32F, DT_CPU | DT_GLTEX );
	m_plot[0].Fill ( 0,0,0,0 );	

	m_plot[1].Resize ( PLOT_RESX, PLOT_RESY, ImageOp::RGBA32F, DT_CPU | DT_GLTEX );
	m_plot[1].Fill ( 0,0,0,0 );	

	m_kernels_loaded = false;

	m_bird_sel = -1;
	
	addSearchPath ( ASSET_PATH );	// <--- so fonts can be found

	init2D ( "arial" );		 // loads the arial.tga font file

	// [optional Start GPU
	if (m_gpu) {
		#ifdef BUILD_CUDA
			cuStart ( DEV_FIRST, 0, m_dev, m_ctx, 0, true );
		#endif
	}

	// Create camera
	m_cam = new Camera3D;
	m_cam->setFov ( 70 );
	m_cam->setNearFar ( 1.0, 100000 );
	m_cam->SetOrbit ( Vec3F(-30,30,0), Vec3F(0,50,0), 300, 1 );

	// Initialize flock simulation
	// 
  
	DefaultParams ();

	m_Params.num_birds = NUM_BIRDS;
	m_Params.num_predators = 0;

	// Initialize experimental setup
	//
	m_run = -1;																				// setup run
	m_num_run = 20;																		// number of samples points
	m_start_frame = 10.0f / m_Params.DT;							// settling time (secs), before measurements start	
	m_end_frame = 40.0f /m_Params.DT + m_start_frame; // end time (secs)

	// tests
	//m_val.Set ( 1000, 11000, 0);						// num_birds						num=20
	//m_val.Set ( 90, 360, 0 );								// field of view				num=27
	//m_val.Set ( 0, 20, 0 );									// boundary cnt					num=20
	//m_val.Set ( 0.001, 0.021, 0 );					// reaction speed				num=40
	//m_val.Set	( 0.001, 0.081, 0);						// avoid_angular_amt		num=32
	//m_val.Set	( 0.020, 0.300, 0);						// reynolds_avoidance		num=28
	//m_val.Set	( 0.100, 2.000, 0);						// align_amt						num=38
	m_val.Set ( 0.05, 2.050, 0 );							// reynolds_alignment		num=40


	m_val.z = float(m_val.y-m_val.x) / m_num_run;
	m_runs_outfile = fopen ( "output.csv", "wt" );	
	fprintf ( m_runs_outfile, "run, num_run, val, #bird, #peaks, peak_ave, peak_max, g0_min,g0_max, g1_min,g1_max, g2_min,g2_max, g3_min,g3_max\n" );

	StartNextRun ();				// this will call Reset
	

	return true;
}

void Sample::drawBackground ()
{
	int w = getWidth(), h = getHeight();
	
	switch ( m_draw_vis ) {
	case 0:
		// realistic sky		
		//drawGradient ( Vec2F(0,0), Vec2F(w,h), Vec4F(.7,.7,.75,1), Vec4F(.7,.7,.75,1), Vec4F(.7,.7,.75,1), Vec4F(.7,.7,.75,1) );
		drawGradient ( Vec2F(0,0), Vec2F(w,h), Vec4F(.6,.7,.8,1), Vec4F(.6,.6,.8,1), Vec4F(1,1,.9,1), Vec4F(1,1,.9,1) );
		break;
	case 1:
		// gray background for vis
		drawFill ( Vec2F(0,0), Vec2F(w,h), Vec4F(.3,.3,.3, 1) );
		break;	
	case 2:
		// white background for publications
		drawFill ( Vec2F(0,0), Vec2F(w,h), Vec4F(1,1,1,1) );
		break;
	};
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
	Predator* p;

	// Advance simulation
	if (m_running) { 		

		for (int i=0; i < m_Params.steps; i++)
			Run ();
	}	


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

	start2D( w, h);			// this 2D draw goes behind (before) the 3D stuff
		drawBackground ();	// <-- this was changed to drawBackground. was SetBackground before.
	end2D();

	start3D(m_cam);			// draws all 3D stuff

		setLight3D ( Vec3F(0, 200, 0), Vec4F(1, 1, 1, 1) );	
		setMaterial ( Vec3F(0,0,0), Vec3F(1,1,1), Vec3F(0,0,0), 40, 1.0 );

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
			drawCircle3D(m_Flock.centroid, 0.5, Vec4F(Vec4F(0.804, 0.961, 0.008, 1)));
		}

		float bird_size = 0.10f; //0.05f;
		float predator_size = 0.1f;

		for (int n=0; n < m_Birds.GetNumElem(0); n++) {

			b = (Bird*) m_Birds.GetElem(0, n);

			// visualization color
			clr = b->clr;
			if ( b->clr.w==0 ) {
				float a = fmin(b->ang_accel.Length() / 48, 1);
				clr = Vec4F( 0, a, 0, 1 );
			}
			if ( m_draw_vis==2) {
				clr = Vec4F( 0, 0, 0, 1);
			}
			
			// bird shape
			if ( m_draw_vis==1 ) {	

				// line
				drawLine3D ( b->pos,		b->pos + (b->vel * bird_size ),	clr );				

			} else {

				// dart
				x = Vec3F(1,0,0) * b->orient;
				y = Vec3F(0,1,0) * b->orient;
				z = Vec3F(0,0,1) * b->orient;
				Vec3F p,q,r,t;
				p = b->pos - z * 0.3f;   // wingspan = 40 cm = 0.2m (per wing)
				q = b->pos + z * 0.3f;
				r = b->pos + x * 0.8f;   // length = 22 cm = 0.22m
				t = y;				
				drawTri3D ( p, q, r, t, clr, true );
			}
		
		}
		// ***** Draw predator with circle around it, static
		Vec4F pclr (1,0,0,1);
		for (int n = 0; n < m_Predators.GetNumElem(FPREDATOR); n++) {
			p = (Predator*)m_Predators.GetElem(FPREDATOR, n);

			//printf("Predator is at: %f, %f, %f \n", p->pos.x, p->pos.y, p->pos.z);

			//pclr = (p->currentState == ATTACK) ? Vec4F(1, 0, 0, 1) : Vec4F(0, 0, 1, 1);				

			drawLine3D (p->pos, p->pos + (p->vel * predator_size), pclr);
			drawCircle3D (p->pos, p->pos + (p->vel * predator_size), 0.5, pclr);
		}
	end3D(); 
	
	Vec4F tc (1,1,1,1);	

	float xscal, yscal;

	start2D ( w, h );			// this 2D draw is overlayed on top
		
		clr = Vec4F(0,0,0,1);			

		// Visualize selected bird
		VisualizeSelectedBird ();

		setTextSz ( 16, 0 );	

		// Spectrum analysis 
		if ( m_draw_plot ) {
			
			// FFT plot
		  drawImg ( &m_plot[0], Vec2F(0,0), Vec2F(PLOT_RESX, PLOT_RESY), Vec4F(1,1,1,1) );

			// FFT fit line
			for (int k=0; k < m_lines.size(); k++) {
				drawLine ( Vec2F(m_lines[k].x, m_lines[k].y), Vec2F(m_lines[k].z, m_lines[k].w), Vec4F(0,0,0,0.3) );
			}
			// FFT energy peaks			
		  for (int k=0; k < m_vis.size(); k++) {
				drawCircle ( m_vis[k].pos, m_vis[k].radius, m_vis[k].clr );				
				drawText ( m_vis[k].pos + Vec3F(0,-16,0), m_vis[k].txt, m_vis[k].clr );
			}			
		}				

		// Energy scatter plot		
		/* xscal = m_Params.max_speed;
		//yscal = (m_method==0) ? 1e-4 : 5e-2;
	  yscal = 50;
		drawRect ( Vec2F( (m_Params.min_speed/xscal)*1000, 600 - 200), Vec2F( (m_Params.max_speed/xscal)*1000, 600), clr );
		for (float v=m_Params.min_speed; v < m_Params.max_speed; v+=1) {		// meters/sec
			drawLine ( Vec2F( (v/xscal)*1000, 600-10), Vec2F( (v/xscal)*1000, 600), clr );		// x-axis ticks
		}
		for (int n=0; n < m_Birds.GetNumElem(0); n++) {
			b = (Bird*) m_Birds.GetElem(0, n);			
			//drawCircle ( Vec2F( (b->speed/xscal)*1000, 600 - (b->Pturn / yscal)*200 ), 1.0, clr);
			drawCircle ( Vec2F( (b->speed/xscal)*1000, 600 - (b->Ptotal / yscal)*200 ), 1.0, clr);
		}  */

		// Graph 
		if ( m_graph.size() > 0 ) {
			Vec2F a,b;			
			// f = t / dt, x = f / 20 = t/(dt*20)
			float tmax = 40.0;									// graph max (seconds)
			
			for (int k=0; k < m_graph.size(); k++) {

				xscal = m_graph[k].scal.x;
				yscal = m_graph[k].scal.y;
				// tick marks
				drawRect ( Vec2F(0, 1200), Vec2F(0+tmax*xscal, 800), clr );
				for (float v = 0; v < tmax; v++) {		// secs
					drawLine ( Vec2F(v*xscal, 1200-10), Vec2F(0+v*xscal, 1200), clr );
				}
				for (float v = 0; v < yscal; v+= yscal/10.0f) {		
					drawLine ( Vec2F(0, 1200-(v/yscal)*400), Vec2F(0+tmax*xscal, 1200-(v/yscal)*400), Vec4F(0,0,0, 0.5) );
				}			
				// plot
				b = Vec3F( 0, 1200 - (m_graph[k].y[0] / yscal)*400, 0);
				//printf ( "%f\n", m_graph[k].y[100] );
				for (int x=0; x < 2084; x++) {
					a = Vec3F( x, 1200 - (m_graph[k].y[x] / yscal)*400, 0);
					drawLine ( a, b, m_graph[k].clr );
					b = a;
				}
			}		
		}

		// Current time
		/* sprintf ( msg, "t = %4.3f sec", m_time );	
		setTextSz ( 24, 0 );						// set text height
		drawText ( Vec2F(getWidth()-600, 10), msg, tc );	*/

	end2D(); 

	drawAll ();				// does actual render of everything
	
	appPostRedisplay();		// Post redisplay since simulation is continuous
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

		//if (m_draw_vis) { 			

			angs.x += dx*0.2f;
			angs.y -= dy*0.2f;				
			m_cam->SetOrbit ( angs, m_cam->getToPos(), m_cam->getOrbitDist(), m_cam->getDolly() );			

		/*} else {		
			angs.x += dx*0.02f;
			angs.y -= dy*0.02f;				
			m_cam->setAngles ( angs.x, angs.y, angs.z );		
		} */

		m_cam_adjust = true;
	} break;	 
	
	};
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
	case 'm': 
		m_method = 1-m_method;
		m_Params.min_speed = (m_method==0) ? 5 : 10;
		Reset ( m_Params.num_birds, m_Params.num_predators);
		break;
	case 'v': 
		if (++m_draw_vis > 2 ) m_draw_vis = 0;
		break;
	case 's': m_draw_sphere = !m_draw_sphere; break;
	case 'g': m_draw_grid = !m_draw_grid; break;
	case 'p': m_draw_plot = !m_draw_plot; break;
  case 'c': 		
		m_cockpit_view = !m_cockpit_view; 
		//m_cam_orient = 
		break;
	case 'r': Reset( m_Params.num_birds, m_Params.num_predators); break;
	case ' ':	m_running = !m_running;	break;	
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
	int w = 2048, h = 1250;
	appStart ( "Flock v2 (c) Rama Karl 2023, MIT license", "Flock v2", w, h, 4, 2, 16, false );	
}

void Sample::shutdown()
{
	// destroy FFTW buffers	
	fftw_destroy_plan( m_fftw_plan);
	fftw_free ( m_fftw_out );
	free ( m_fftw_in );
}


