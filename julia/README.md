# Documentation
- This directory contains a Julia package called AutoEnvs implementing the environments
- It also contains tests for those environments

## AutoEnvs
- This is the julia package implementing the driving environments
- There are three main environments

### NGSIMEnv
- This is the simplest environment
- It injects a single agent into an otherwise predetermined set of NGSIM trajectories

### VectorizedNGSIMEnv
- This is a vectorized version of NGSIMEnv
- This means that many agents are executed simultaneous in _separate_ environments, but just like in NGSIMEnv
- This environment exists to improve efficiency 
  + Communication between python and julia is costly
  + Running individual forward props through networks is slower than batch forward props, particularly when running with a GPU
  + This environments groups together agent info so that fewer communication steps are required, and all the forward props are performed in batch

### MultiagentNGSIMEnv
- This environment allows for training multiple agents in the same environment simultaneously
- The way it does this is by acting as though it were a vectorized environment
  + It kind of is, in the sense that it executes many agents at once
  + And it kind of is not in that only a single environment is executing at once
- _This is a simple implementation of multiple agents_
  + The reason for this is that it assumes _synchronous_ resets across agents
  + This means that it is restricted to simulating agents in groups such that all agents in the group can start and stop at the same timestep
  + This means you can't replace all the trajectories in the environment with GAIL, only a subset
  + It was implemented this way because it made the environment simpler to implement

## Tests
- Run `julia runtests.jl` to execute the tests
- This may take a few minutes
- If it does not print out errors, then everything works
- It will also display timing information as well as some test output
