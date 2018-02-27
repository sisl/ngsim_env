# Documentation
- This directory contains the python code that interfaces between the julia environment and (typically) rllab samplers
- See `julia_env/julia_env.py` for the main part of the code
- It creates a Julia interpreter in its constructor
- It defines the rllab/gym functions for an environment (reset, step, render), and passes arguments along to julia
