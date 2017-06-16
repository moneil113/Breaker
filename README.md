# Breaker Fluid Simulation

Breaker is a smooth particle hydrodynamics fluid simulation, modeling a fluid as
many thousands of discrete particles. The simulation exists in two forms: a
multithreaded CPU implementation using the C++ `std::thread` library, and a GPU
implementation written in [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone).


Dependencies
* [GLFW](http://www.glfw.org/)
* [GLEW](http://glew.sourceforge.net/)
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

To build and run the GPU branch, your computer must have a CUDA capable
NVIDA GPU.

## Build instructions

To build Breaker, first clone this repository and install the above libraries.
I recommend an out of source build, so create a build directory in `Breaker/`
After setting the environment variables `EIGEN3_INCLUDE_DIR`, `GLFW_DIR`, and
`GLEW_DIR`, simply run `cmake ..` from the build directory. This will generate
the necessary makefiles to build Breaker. Run `make` to compile the source files.

## Running Breaker

Breaker requires as a command line argument the path to a resource directory
with GLSL shaders called for rendering the particles and domain boundaries.
`./Breaker ../resources` will run the program with default shaders.

While the program is running, click and drag to rotate the camera. Shift + click
to pan, and ctrl + click to zoom. Press `space` to pause/continue
simulation, and press `R` to cycle through a number of particle spawning
behaviors. `G` toggles gravity, and `F` freezes the velocity of all particles.
