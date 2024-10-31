# SPECULA
Python AO end-to-end simulator

SPECULA is a python based object oriented software developed
in the Adaptive Optics group of the Arcetri observatory for Monte-Carlo end-to-end adaptive optics simulations.
It can be accelerated with GPU-CUDA relying on CuPy.

Directories:
* calib: this directory contains functions/classes to make the calibration of an adaptive optics system
* classes: this directory contains classes files (Data objects, Processing objects and Management/Coordination objects)
* cloop: this directory contains functions/classes to run a closed loop of a adaptive optics system (single-conjugated, multi-conjugated, natural, laser, ...)
* error_budget: this directory contains (semi-analytical) functions/classes to estimate the error budget of a single-conjugated adaptive optics system
* gkl: this directory contain a collection of routines related to Karhunen-Loeve modes - reference: Robert C. Cannon, "Optimal bases for wave-front simulation and
reconstruction on annular apertures", Journal of the Optical Society of America A, Vol. 13, Issue 4, pp. 862-867 (1996), https://doi.org/10.1364/JOSAA.13.000862
* lib: this directory contains functions/procedure
* main: this directory contains a few examples of main and parameters files to calibrate and run some adaptive optics systems
* test: this directory contains a set of functions to test SPECULA using make_test.pro file
* test_gpu: this directory contains a set of functions to test SPECULA with the GPU dll using make_test.pro file

Libraries required:
* 
* 
* 

There is a help doc [help.pdf](help.pdf) for this software.

The reference publication for this software is:








