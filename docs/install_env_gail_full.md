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
conda install hdf5
conda install matplotlib
# Check the install went through correctly
python
    >>>import mpl_toolkits
    >>>quit()

# activate the rllab environment
source activate rllab3
python setup.py develop
cd ..
```

## Install julia
Install julia 1.1. See the internet for instructions. Make sure the `julia` command pops up a julia 1.1 interpreter. More detailed instructions coming later.

## Install ngsim_env

### julia
```bash
source activate rllab3
git clone https://github.com/sisl/ngsim_env.git
sudo apt-get install libgtk-3-dev
#   NOTE: If you do not have sudo access, you can probably get away with not doing this, there just may be an error when adding AutoViz.

# enter a julia interpreter and install dependencies.
#   NOTE: I got some weird error with one of the packages, I think it was AutoViz, where there was a line ending before expected or something like that.
#   I just repeated the add instruction and it seemed to work fine.
julia
  # Add dependences
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

  # NOTE - I did not have to do this on a new install on a remote server.
  # set python path (replace with your miniconda3 install location)
  >>ENV["PYTHON"] = "/home/<your_username_here>/miniconda3/envs/rllab3/bin/python"
  >>using Pkg
  >>Pkg.build("PyCall")
  >>Pkg.build("PyPlot")
  >>using PyCall
  >>using PyPlot
  >>Pkg.build("HDF5")
  >> quit()
  # If it doesn't work immediately, I got an error saying extra trailing apt, restart the terminal and try again
```

Next, we will get the NGSIM data and run a few tests with julia and python to make sure everything is fine

### download NGSIM data
```bash
##Get the data
cd ~/.julia/packages/NGSIM/B45UX/data
wget https://github.com/sisl/NGSIM.jl/releases/download/v1.0.0/data.zip
unzip data.zip

##Create trajectories from the data
cd ../data
julia
  >> using NGSIM
  >> convert_raw_ngsim_to_trajdatas()
  >> quit()
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

### run python tests
```bash
# install the python components of the package
source activate rllab3
cd ~/ngsim_env/python # this is assuming you have ngsim_env on your home directory. If not, navigate to where you have ngsim_env
python setup.py develop
conda install julia
cd tests
# NTOE: one of the these tests will fail if you don't have hgail installed
python runtests.py

# After removing PyCall.jl let's try the test again (all this is assuming you got a seg fault)
cd ~/ngsim_env/python/tests
python runtests.py
  # If you get the error: 
  # ERROR: test_vectorized_ngsim_env (unittest.loader._FailedTest)
  # Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so

  conda install nomkl numpy scipy scikit-learn numexpr
  # Found the fix https://stackoverflow.com/questions/36659453/intel-mkl-fatal-error-cannot-load-libmkl-avx2-so-or-libmkl-def-so
  # Answer by libphy
  # Then run the test again and it should be fine

```

# Installation instructions for the imitation learning algorithm
```bash
cd ~
git clone https://github.com/sisl/hgail.git
source activate rllab3
cd hgail
python setup.py develop
cd tests python 
python runtests.py
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
