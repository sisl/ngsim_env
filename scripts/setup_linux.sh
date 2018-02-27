# preamble
echo "Hello, this is an install script for ngsim_env"
echo "Do not run this as sisl@vestavia"
echo ""
echo "make sure you run this script somewhere you don't mind installing a bunch of stuff"
echo "happy with the current directory? [y|n]"
read continue
if [ "$continue" != "y" ]
then
    echo "ok, go to some better directory then"
    exit
else
    echo "ok, installing here"
fi

# python install
echo "Installing python stuff..."
## install miniconda
echo "Installing miniconda..."
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

## rllab
echo "Installing rllab..."
### dependencies
pip install pygame, scipy, matplotlib, pyjulia
git clone https://github.com/rll/rllab.git
cd rllab
echo "Creating rllab conda environment..."
echo "This will take a while..."
conda env create -f environment.yml
conda env update
python setup.py develop
echo "Conda environment created! Make sure to run \`source activate rllab3\` whenever you open a new terminal and want to run programs under rllab."
cd ..
source activate rllab3
pip install julia

# julia install 
echo "Installing julia stuff..."

wget https://julialang-s3.julialang.org/bin/linux/x64/0.6/julia-0.6.2-linux-x86_64.tar.gz
tar -xf julia-0.6.2-linux-x86_64.tar.gz
rm julia-0.6.2-linux-x86_64.tar.gz

echo "Adding some stuff we'll need later to bashrc..."
echo "export LD_PRELOAD=${HOME}/.julia/v0.6/Conda/deps/usr/lib/libz.so" >> ~/.bashrc 
echo "export PATH=$(pwd)/julia-d386e40c17/bin/:$PATH" >> ~/.bashrc

## automotive packages
echo "Installing automotive stuff..."
### julia part of ngsim_env
git clone https://github.com/wulfebw/ngsim_env.git
julia ngsim_env/julia/deps/build.jl
echo "Adding AutoEnv to julia path in juliarc.jl"
echo "push!(LOAD_PATH, \"$(pwd)/ngsim_env/julia/AutoEnvs\")" >> ~/.juliarc.jl
### automotive python setup
cd ngsim_env/python
python setup.py develop
cd ../..

### NGSIM
echo "At this point you need to download NGSIM data"
echo "see the julia package NGSIM.jl for instructions"
echo "After you do that, run the julia tests"
echo "Then after those work, get access to the GAIL implementation"
echo "And run the imitate.py script in the scripts directory"

