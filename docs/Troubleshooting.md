# Some errors are likely to arise.

We didn't run into too many while doing this, but there was an issue when generating validation files for the first time for a give H(orizon), 
as it has to create index files and things. 
The problem was that it would segfault while running validate.
The first step of the solution was the same as is included in the install guide:

    cd ~/.julia/lib/v0.6
    rm PyCall.jl
    # Check that it works
    python
      >>>import julia
      >>>quit()
      
### If the problem persists when running validate.py, see below.

After that, its basically running validate.py once for each time period (changing the files hardcoded in the file). 
After the first time validate is run, files like:
    (in ~/.julia/v0.6/NGSIM/data/):

  trajdata_i80_trajectories-0515-0530-index-250-ids.h5
  trajdata_i80_trajectories-0515-0530-index-250.jld
  trajdata_i80_trajectories-0515-0530-index-250-starts.h5
  
will have been created. Subsequent runs should work fine.  
The 250 here corresponds to the desired horizon (200) plus an offset we use (50). 

It is recommended to do these runs with the --multiagent_env False flag (it is default), since single agent val needs some of the files that multiagent doesnt.
