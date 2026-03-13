Generative Thermodynamic Computing

This code implements the generative modeling framework described in “Generative Thermodynamic Computing” (https://arxiv.org/abs/2506.15121) by Stephen Whitelam (2025). It simulates a nonequilibrium thermodynamic system that learns to generate structured data from noise using physical dynamics alone.


⸻

What the Code Does

The code simulates a collection of interacting units, some visible (used for displaying data) and some hidden (used for computation), whose states evolve in time due to internal forces and thermal noise.

Training proceeds by maximizing the probability that the system could have reversed a noising trajectory. This corresponds to minimizing the heat emitted during generation.

The simulation proceeds in the following stages:
 1. Data loading: Three MNIST digits are read and used for training.
 2. Noising: The system starts from a chosen digit and evolves toward noise to generate training data. Output: `report_all_noising_trajectories.png`
 3. Training: The system is trained to run this process in reverse, generating digits from noise, by adjusting internal couplings and biases via gradient descent.
 4. Visualization: After training, the system runs autonomously from noise and generates a variety of digit-like outputs (report_assess_denoised_digits.png, report_all_denoising_trajectories.png)
 5. Analysis: report_connection_fields.png shows some hidden-visible connections

⸻

How to Build and Run

Requirements:
 • C++11 compiler (g++, clang++)
 • ImageMagick (for montage and PNG output)
 • zlib (used only for minimal PNG writing fallback)

Build and Run with

 g++ -Wall -o sim gen.c -lm -O; ./sim
 
 Input
 • mnist_data.dat — plaintext file with 3 MNIST digits. Each digit has a label followed by 784 (scaled) grayscale values. This is the only required input.

⸻

Outputs
 • report_all_noising_trajectories.png — visualization of the noising process
 • report_loss.dat — training loss per cycle
 • parameters.dat — learned model parameters
 • report_connection_fields.png — learned coupling patterns from hidden units
 • report_all_denoising_trajectories.png — 10 trajectories generated from noise
 • report_assessed_denoised_digits.png — 100 final samples (at t=tf) from the trained model

⸻

Key Functions in the Code

Initialization
 • initialize() — seeds RNG, clears output files, reads digit data, sets up connections
 • read_mnist() — loads training digits from file

Dynamics
 • langevin_step() — performs one time step of Langevin dynamics
 • run_trajectory() — evolves the system over time and records snapshots if requested

Training
 • train_computer() — performs gradient-based training to adjust couplings and biases
 • accumulate_gradients() — computes contributions to the training gradient

Data Preparation
 • set_initial_image_data(int) — loads a digit to start noising
 • set_initial_noise_data() — initializes the system with noise prior to generation

Visualization
 • make_lattice_picture(int) — renders current visible layer as an image
 • example_noising_trajectories() — runs noising trajectories for the training digits
 • example_denoising_trajectories() — runs 10 denoising trajectories from noise (output: `report_all_denoising_trajectories.png`)
 • assess_denoised_digits() — runs 100 denoising trajectories and saves final-time output (output: `report_assessed_denoised_digits.png`)
 • connection_field_grid() — plots connection patterns from hidden units to the display layer

⸻


## Citation

Whitelam, *Generative Thermodynamic Computing*, 2025  
https://arxiv.org/abs/2506.15121
