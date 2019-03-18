# Installing ngsim_env and gail
- These are the install instructions for the environment which is called ngsim_env followed by generative adversarial imitation learning called hgail

# Installation instructions for ngsim_env 
```bash
# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# answer yes to everything
sh ./Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# install rllab
git clone https://github.com/rll/rllab.git
cd rllab
# this takes a while
conda env create -f environment.yml
conda env update
# this may give an error about twisted, tensorflow dpeending on a different numpy version. Ignore it for now.
# If hdf5 is not installed, install it as it is required by AutoEnvs later in the process
sudo apt-get install hdf5-tools
conda activate rllab3
# TRYING WITHOUT: pip install hdf5
# TRYING WITHOUT: pip install matplotlib
# Check the install went through correctly
python
    import numpy as np
    import tensorflow as tf
    import mpl_toolkits
    quit()
# NOTE: I had some issues with tensorflow upgrading to v 2.0, which as you may imagine, breaks things.
# If you encounter this issue, there may be some manual downgrading / reinstalling of tensorflow and numpy required.
# I am testting with downgrading tensorflow.
conda activate rllab3
pip install tensorflow==1.12.0
pip install numpy==1.15.4
# then test the above again



# activate the rllab environment
source activate rllab3
python setup.py develop
cd ..
```

## Install julia
Install julia 1.1. Code snippet assuming you are installing julia to home directory. If not, please 
modify the path in bashrc step accordingly.

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz
tar -xvf julia-1.1.0-linux-x86_64.tar.gz
rm julia-1.1.0-linux-x86_64.tar.gz
echo "export PATH=$(pwd)/julia-1.1.0/bin:$PATH" >> ~/.bashrc
```
Make sure the `julia` command pops up a julia 1.1 interpreter.

## Install ngsim_env

### julia
```bash
source activate rllab3
git clone https://github.com/sisl/ngsim_env.git
cd ngsim_env
git checkout 0.7fixes_in_progress
sudo apt-get install libgtk-3-dev
#   NOTE: If you do not have sudo access, you can probably get away with not doing this, there just may be an error when adding AutoViz.

# enter a julia interpreter and install dependencies.
#   NOTE: I got some weird error with one of the packages, I think it was AutoViz, where there was a line ending before expected or something like that.
#   I just repeated the add instruction and it seemed to work fine.
julia
  # Add dependencesjulia
  using Pkg
  Pkg.add(PackageSpec(url="https://github.com/sisl/Vec.jl"))
  Pkg.add(PackageSpec(url="https://github.com/sisl/Records.jl"))
  Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveDrivingModels.jl"))
  Pkg.add(PackageSpec(url="https://github.com/sisl/AutoViz.jl"))
  Pkg.add(PackageSpec(url="https://github.com/sisl/AutoRisk.jl.git", rev="v0.7fixes"))
  Pkg.add(PackageSpec(url="https://github.com/sisl/NGSIM.jl.git"))
  Pkg.add(PackageSpec(url="https://github.com/sisl/BayesNets.jl.git"))

  # Add the local AutoEnvs module to our julia environment
  ] dev ~/ngsim_env/julia
 
  # make sure it works
  using AutoEnvs
```

Now, we want to make sure we add the other files we care about.
```
julia deps/build.jl

# make sure they work, by entering interpreter

julia
  using PyCall
  using PyPlot
  using HDF5
  quit()
  
# If using PyPlot errors saying "No module named matplotlib") do the following
(rllab3)>> conda install matplotlib
julia
    using PyCall
    using PyPlot
```

Next, we will get the NGSIM data and run a few tests with julia and python to make sure everything is fine

### download NGSIM data
```bash
##Get the data
cd ~/.julia/packages/NGSIM/B45UX/data
wget https://github.com/sisl/NGSIM.jl/releases/download/v1.0.0/data.zip
unzip data.zip
# Answer yes to any that ask to be replaced.

##Create trajectories from the data
julia
  >> using NGSIM
  >> convert_raw_ngsim_to_trajdatas()
  >> quit()
# NOTE: my attempt got killed here and i have no idea why. No error messages or anything.
```

### run julia tests

```bash
# run the julia tests
# if you don't get an error, everything works with julia
# it will take a few minutes because it's creating some cached files
cd ~/ngsim_env/julia/test
julia runtests.jl
cd ~
```
### Install the imitation learning algorithm gail and test it
```bash
# NOTE: One of the tests fails for me across multiple installs. "test_train_domain_matters"
cd ~
git clone https://github.com/sisl/hgail.git
source activate rllab3
cd hgail
python setup.py develop
cd tests 
python runtests.py
```

### Test ngsim_env python stuff
```bash
# install the python components of the package
source activate rllab3
cd ~/ngsim_env/python # this is assuming you have ngsim_env on your home directory. If not, navigate to where you have ngsim_env
python setup.py develop
pip install julia
# make sure it works:
python
  import julia
  julia.Julia()
  # we just want to make sure it doesnt error
  quit()

cd tests
# NOTE: one of the these tests will fail if you don't have hgail installed
python runtests.py
#   # NOTE: If you get the error: 
#   # ERROR: test_vectorized_ngsim_env (unittest.loader._FailedTest)
#   # Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so
#   conda install nomkl numpy scipy scikit-learn numexpr
#   # found the fix https://stackoverflow.com/questions/36659453/intel-mkl-fatal-error-cannot-load-libmkl-avx2-so-or-libmkl-def-so
#   # Answer by libphy
#   # Then run the test again and it should be fine

```

# Extract demonstration data from NGSIM
```bash
cd ~/ngsim_env
mkdir data
mkdir data/trajectories
mkdir data/experiments
cd ~/ngsim_env/scripts

julia extract_ngsim_demonstrations.jl
#code references both ngsim.h5 and ngsim_all.h5, so make a copy?
cd ../data/trajectories
cp ngsim_all.h5 ngsim.h5
```
Congratulations!! You have completed the installation process. Navigate back to main [readme](https://github.com/sisl/ngsim_env/blob/master/README.md)
page and look at the 'Train and run a single agent GAIL policy:' section to train a policy

*this was originally compiled by ![raunakbh92](https://github.com/raunakbh92/InstallInstructions/edit/master/install_ngsim_env_hgail.md)*
