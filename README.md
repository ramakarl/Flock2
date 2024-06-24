<img src="https://github.com/ramakarl/Flock2/blob/main/docs/fig_flock2.jpg" width="800" />

## Flock2: A model for orientation-based social flocking

This repository holds the source code for the paper:<br>
Hoetzlein, Rama. "Flock2: A model for orientation-based social flocking", Journal of Theoretical Biology, 2024<br>

### Supplementary Video
Supplementary video for the paper can be found here:
[Flock2: Youtube video](https://www.youtube.com/watch?v=lDEXNLLCwRU)

### Abstract
The aerial flocking of birds, or murmurations, have fascinated observers while presenting many challenges to behavioral study and simulation. We examine how the periphery of murmurations remain well bounded and cohesive. We also investigate agitation waves, which occur when a flock is disturbed, developing a plausible model for how they might emerge spontaneously. To understand these behaviors a new model is presented for orientation-based social flocking. Previous methods model inter-bird dynamics by considering
the neighborhood around each bird, and introducing forces for avoidance, alignment, and cohesion as three dimensional vectors that alter acceleration. Our method introduces orientation-based social flocking that treats social influences from neighbors more realistically as a desire to turn, indirectly controlling the heading in an aerodynamic model. While our model can be applied to any flocking social bird we simulate flocks of starlings, Sturnus vulgaris, and demonstrate the possibility of orientation waves in the absenc of predators. Our model exhibits spherical and ovoidal flock shapes matching observation. Comparisons of our model to Reynolds' on energy consumption and frequency analysis demonstrates more realistic motions, significantly less energy use in turning, and a plausible mechanism for emergent orientation waves.

### Model Overview
<img src="https://github.com/ramakarl/Flock2/blob/main/docs/fig_models.png" width="800" />

Flock2 is the first orientation-based controller for collective flocking. The classical model by a) Craig Reynolds (1987) implements the social factors for alignment, avoidance and cohesion as 3D vector forces. b) Hildenbrandt et al. (2010) also introduce an aerodynamic model, however, the social forces do not influence the flight forces in their work as they are uncoupled. c) In Flock2 we introduce a high-level social orientation model for heading which directly controls a low-level aerodynamic force model.

Flock2 uses a *perceptual* model of social factors which projects the avoidance, alignment, cohesion influences from nearest neighbors onto the visual sphere of each bird, and then derives a target heading to strictly control an aerodynamic force model. We demonstrate that a flight model directly controlled by a 2D orientation-based high level controller is still able to produce flocking behaviors, including novel features such as emergent orientation waves not previously seen.  

### Aerodynamic Model
Flock2 contains a low-level controller to model the Aerodynamic Forces of each bird. The low-level controller, [Flightsim](https://github.com/ramakarl/flightsim) (https://github.com/ramakarl/flightsim), is fully integrated into Flock2 and does not need to be compiled separately. This aerodynamic model includes lift, gravity and drag, and also exhibits features such as altitude loss during banking, speed changes when diving or climbing, angle-of-attack and stalls. Flightsim is a fixed-wing, single-body force model (SBFM) based on dynamic stability, which is more efficient and less complex that a full aircraft model that iterates over multiple control surfaces. See more details on the motivation and design of [Flightsim here](https://github.com/ramakarl/flightsim). 

### Quick Installation
Platforms: Windows or Linux<br>
Steps:<br>
1. Clone this repository locally.<br>
2. Install Git (cmdline), Cmake and Visual Studio 2019 or higher.
3. For GPU support (NVIDIA only), install CUDA Toolkit 10.2 or higher.<br>
4. Run **build_all.bat** (Windows) or **build_all.sh** (Linux) for your platform.<br>

### Rama Carl Hoetzlein (c) 2023-2024. MIT License
[https://ramakarl.com](https://ramakarl.com)


