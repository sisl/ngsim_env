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
# If hdf5 is not installed, install it as it is required by AutoEnvs later in the process
conda install hdf5
# activate the rllab environment
source activate rllab3
python setup.py develop
cd ..
```

## Install julia
```bash
source deactivate
# install our own version of julia
wget https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.2-linux-x86_64.tar.gz
tar -xf julia-0.6.2-linux-x86_64.tar.gz
rm julia-0.6.2-linux-x86_64.tar.gz

# add this line to avoid a pyjulia issue where it uses the wrong julia
# this makes it so that by default we use this julia install
echo "export PATH=$(pwd)/julia-d386e40c17/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

## ngsim_env

### julia
```bash
source activate rllab3 # this is probably not necessary, but just in case
git clone https://github.com/sisl/ngsim_env.git
# this takes a long time
julia ngsim_env/julia/deps/build.jl

# this avoids a bug
echo "using Iterators" >> ~/.juliarc.jl
# avoiding a PyPlot bug
echo "export LD_PRELOAD=${HOME}/.julia/v0.6/Conda/deps/usr/lib/libz.so" >> ~/.bashrc 
source ~/.bashrc
# manually add AutoEnvs
echo "push!(LOAD_PATH, \"$(pwd)/ngsim_env/julia/AutoEnvs\")" >> ~/.juliarc.jl

# Revert to a previous version of Vec.jl
cd ~/.julia/v0.6/Vec
git fetch --tags
git checkout v0.1.0

# Revert  AutomotiveDrivingModels to commit before update for Vec.jl
cd ~/.julia/v0.6/AutomotiveDrivingModels
git checkout 74050e9ae44bda72a485c2573ac4f0df2bc3e767

# enter a julia interpreter
julia
  # set python path (replace with your miniconda3 install location)
  >>ENV["PYTHON"] = "/home/wulfebw2/miniconda3/envs/rllab3/bin/python"
  # if any of this raises a bug, fix it before moving on
  # this installs the julia-internal conda referenced above
  >>using PyCall
  >>using PyPlot
    # takes a while
      # If this errors ImportError('No module named mpl_toolkits.mplot3d',), you need to upgrade matplotlib
  >>quit()
# The following steps (until and including python quit()) are only required if julia gives 
# an import error while using PyPlot
pip install --upgrade matplotlib
# Check the install went through correctly
python
    >>>import mpl_toolkits
    >>>quit()

# Open up Julia interpreter again and try using PyPlot again
julia
  >> using PyPlot
  >>using AutoEnvs
  # If this AutoEnvs step errors saying problems with HDF5, just do what it suggests
  >> Pkg.build("HDF5")
  # If the above errors, do what is says i.e. sudo apt-get install hdf5-tools
  >> quit()
sudo apt-get install hdf5-tools
  # If it doesn't work immediately, I got an error saying extra trailing apt, restart the terminal and try again
  # Now let's go back to Julia and try using AutoEnvs again
julia
  >> using AutoEnvs
  >> quit()
```
Next, we will get the NGSIM data and run a few tests with julia and python to make sure everything is fine

### download NGSIM data
```bash
##Get the data
cd ~/.julia/v0.6/NGSIM/data
wget https://github.com/sisl/NGSIM.jl/releases/download/v1.0.0/data.zip
unzip data.zip

##Fix a trajectory converstion script
cd ../src
#change line 205 of .julia/v0.6/NGSIM/src/trajdata.jl" to outpath = Pkg.dir("NGSIM", "data", "trajdata_"*splitdir(filename)[2])

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
cd ngsim_env/julia/test
julia runtests.jl
cd ../../..
```

### run python tests
```bash
# install the python components of the package
source activate rllab3
cd ngsim_env/python # this is assuming you have ngsim_env on your home directory. If not, navigate to where you have ngsim_env
python setup.py develop
pip install julia
cd tests
# one of the these tests will fail if you don't have hgail installed

python runtests.py
  # if you get a segfault, need to delete the PyCall.jl cache file

  cd ~/.julia/lib/v0.6
  rm PyCall.jl
  # Check
  python
    >>>import julia
    >>>quit()

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
cd tests python runtests.py
cd ~/ngsim_env
mkdir data
mkdir data/trajectories
mkdir data/experiments
cd scripts
julia
  >> Pkg.checkout("AutomotiveDrivingModels", "lidar_sensor_optimization")
  >> quit()

cd ~/ngsim_env/scripts

julia extract_ngsim_demonstrations.jl
#code references both ngsim.h5 and ngsim_all.h5, so make a copy?
cd ../data/trajectories
cp ngsim_all.h5 ngsim.h5
```
Congratulations!! You have completed the installation process. Navigate back to main [readme](https://github.com/sisl/ngsim_env/blob/master/README.md)
page and look at the 'Train and run a single agent GAIL policy:' section to train a policy

*this was originally compiled by ![raunakbh92](https://github.com/raunakbh92/InstallInstructions/edit/master/install_ngsim_env_hgail.md)*
