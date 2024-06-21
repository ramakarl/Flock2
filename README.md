<img src="https://github.com/ramakarl/Flock2/blob/main/docs/fig_flock2.jpg" width="800" />

## Flock2: A model for orientation-based social flocking

This repository holds the source code for the paper:<br>
Hoetzlein, Rama. "Flock2: A model for orientation-based social flocking", Journal of Theoretical Biology, 2024<br>

### Supplementary Video
Supplementary video for the paper can be found here:
[Flock2: Youtube video](https://www.youtube.com/watch?v=lDEXNLLCwRU)

### Abstract
The aerial flocking of birds, or murmurations, have fascinated observers while presenting many challenges to behavioral study and simulation. We examine how the periphery of murmurations remain well bounded and cohesive. We also investigate agitation waves, which occur when a ock is disturbed, developing a plausible model for how they might emerge spontaneously. To understand these behaviors a new model is presented for orientation-based social flocking. Previous methods model inter-bird dynamics by considering
the neighborhood around each bird, and introducing forces for avoidance, alignment, and cohesion as three dimensional vectors that alter acceleration. Our method introduces orientation-based social flocking that treats social influences from neighbors more realistically as a desire to turn, indirectly controlling the heading in an aerodynamic model. While our model can be applied to any ocking social bird we simulate ocks of starlings, Sturnus vulgaris, and demonstrate the possibility of orientation waves in the absenc of predators. Our model exhibits spherical and ovoidal flock shapes matching observation. Comparisons of our model to Reynolds' on energy consumption and frequency analysis demonstrates more realistic motions, signicantly less energy use in turning, and a plausible mechanism for emergent orientation waves.

### Model Overview
<img src="https://github.com/ramakarl/Flock2/blob/main/docs/fig_model.png" width="800" />

Flock2 is the first orientation-based controller for collective flocking. The classical model by a) Craig Reynolds (1987) implements the social factors for alignment, avoidance and cohesion as 3D vector forces. b) Hildenbrandt et al. (2010) also introduce an aerodynamic model, however, the social forces do not influence the flight forces in their work as they are uncoupled. c) In Flock2 we introduce a *perceptual* model of social factors which projects the influence from nearest neighbors onto the visual sphere of each bird, and uses a target heading to strictly control an aerodynamic model. We demonstrate that a flight model strictly controlled by a 2D orientation-based high level controller is still able to produce flocking behaviors, including novel features such as emergent orientation waves not previously seen.  

### Rama Carl Hoetzlein (c) 2023-2024. MIT License
[https://ramakarl.com](https://ramakarl.com)


