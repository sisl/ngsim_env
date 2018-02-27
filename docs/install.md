
# Install
- these are instructions because I haven't gotten an install script to work yet
    + see `scripts/setup_linux.sh` for an attempt at it
    + that'd be cool if someone could get it working
- some of the commands below avoid hard to debug bugs, so don't advise skipping any of them

## python
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
# activate the conda environment
source activate rllab3
python setup.py develop
cd ..
```

## julia
```bash
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
git clone https://github.com/wulfebw/ngsim_env.git
# this takes a long time
julia ngsim_env/julia/deps/build.jl

# this avoids a bug
echo "using Iterators" >> ~/.juliarc.jl
# avoiding a PyPlot bug
echo "export LD_PRELOAD=${HOME}/.julia/v0.6/Conda/deps/usr/lib/libz.so" >> ~/.bashrc 
source ~/.bashrc
# manually add AutoEnvs
echo "push!(LOAD_PATH, \"$(pwd)/ngsim_env/julia/AutoEnvs\")" >> ~/.juliarc.jl
source activate rllab3
# enter a julia interpreter
julia
# set python path (replace with your miniconda3 install location)
ENV["PYTHON"] = "/home/wulfebw2/miniconda3/envs/rllab3/bin/python"
# if any of this raises a bug, fix it before moving on
# this installs the julia-internal conda referenced above
using PyCall
using PyPlot
# takes a while
using AutoEnvs
quit()
```

### download NGSIM data
- see [NGSIM.jl](https://github.com/sisl/NGSIM.jl) for instructions

### run julia tests

```bash
# run the julia tests
# if you don't get an error, everything works with julia
# it will take a few minutes because it's creating some cached files
cd ngsim_env/julia/test
julia runtests.jl
cd ../../..
```

### python
```bash
# install the python components of the package
source activate rllab3 # this is probably not necessary, but just in case
cd ngsim_env/python
python setup.py develop
pip install julia
cd tests
# one of the these tests will fail if you don't have hgail installed
# if you get a segfault, see below (PyCall cache)
python runtests.py
```

# Troubleshooting

### pyjulia

#### segfault 
##### reason 1: PyCall cache 
- this is due to a bug with the PyCall cache file 
- this file is stored in .julia/lib/<pyjulia-or-something>
    + it's also stored in .julia/lib/<v0.6-or-something>
- to fix the bug you have to delete the PyCall cache file called PyCall.ji from both caches, and then in python import julia
    + don't delete the full cache because then it takes forever to rebuild, just delete PyCall.ji

##### reason 2: julia version mixup
- on tver for example the default julia is julia v0.5, but you need v0.6
- so what you have to do is source whatever conda env you're using 
- then manually export the path to the julia binary to be _first in the path_
- and then install via pip install julia 
- and then build
- you should only have to do this once though

#### iterators uuid messed up
- no idea what the problem is
- but if you import Iterators in the juliarc.jl file it fixes it 
- i.e., add `using Iterators` to the top of `~/juliarc.jl`