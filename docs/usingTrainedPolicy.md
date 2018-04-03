# Instructions to generate a simulation video given a trained policy
Note before we start: Raunak had cloned the ngsim_env repo on his home directory (i.e. `~`). The following instructions
are written with that in mind. So for your own case, please accordingly navigate to your location of ngsim_env

```bash
# get updated version of repository
cd ~/nsgim_env
git pull

# If you have not already, create the directories where the policies will be kept
# If you already have these directories setup and random_model_name as the experiment name already setup
# please navigate ahead to the instructions for creating a directory called media and continue from there
mkdir data/
mkdir data/experiments

# Caution: If you want to change the experiment name which is random_model_name in the next few instructions
# you will need to change the name in the result generation files later as well
# I suggest you keep it as it is
mkdir data/experiments/random_model_name
mkdir data/experiments/random_model_name/imitate
mkdir data/experiments/random_model_name/imitate/log

# Now copy the trained policy you already have into the directory created above
cd data/experiments/random_model_name/imitate/log/
# Then copy the policy files into this directory. These are args.npz, iter_200.npz, log.txt

# Now create the media directory where the resulting video will be generated
mkdir ~/ngsim_env/data/media

# revert a breaking change to the Vec.jl dependency by going into the julia packages directory
cd ~/.julia/v0.6/Vec
git checkout v0.1.0

# Now install some things (see errors section below to know the specific errors these address)
cd ~/ngsim_env
source activate rllab3
pip install absl-py
pip install contexttimer
pip install theano
pip install pyprind
pip install tensorflow
conda install mkl-service
conda install mkl=2017
sudo apt-get install ffmpeg

# Now we are ready to run the policy
cd ~/ngsim_env/scripts/imitation
# This assumes you have jupyter notebook installed. If not, install it using
sudo apt-get -y install ipython ipython-notebook
sudo -H pip install jupyter

# Now activate the rllab environment and open up a jupyter notebook
jupyter notebook

# This will open up a notebook on your browser and you will see all the files present in the
# current directory listed on the browser

# Open visualize_trajectories-SIMPLE
# Run the cells one by one
# The last cell is the one that generates the video by running the trained policy
# Possible error within visualize_trajectories-SIMPLE
SystemError: Julia exception: SystemError("mkdir", 2, "/tmp/imitate/viz")
# Fix by doing the following
mkdir /tmp/imitate
mkdir /tmp/imitate/viz

# If the run goes succesfully, you will see a video file in ~/ngsim_env/data/media
# This is called random_model_name_0_0.mp4 
```

## Errors Raunak found and their associated fixes
- absl
```bash
ImportError: No module named 'absl'
(rllab3) raunak@bombay:~/ngsim_env/scripts/imitation$ pip install absl-py
```
- contexttimer
```bash
ImportError: No module named 'contexttimer'
(rllab3) raunak@bombay:~/ngsim_env/scripts/imitation$ pip install contexttimer
```

- theano
```bash
ImportError: No module named 'theano'
(rllab3) raunak@bombay:~/ngsim_env/scripts/imitation$ pip install theano
```

- pyprind
```bash
ImportError: No module named 'pyprind'
(rllab3) raunak@bombay:~/ngsim_env/scripts/imitation$ pip install pyprind
```

- No module named 'mkl'
```bash
WARNING (theano.configdefaults): install mkl with `conda install mkl-service`: No module named 'mkl'
Segmentation fault (core dumped)
(rllab3) raunak@bombay:~/ngsim_env/scripts/imitation$ conda install mkl-service
```

- "MKL_THREADING_LAYER=GNU"
```bash
RuntimeError: To use MKL 2018 with Theano you MUST set "MKL_THREADING_LAYER=GNU" in your environement.
(rllab3) raunak@bombay:~/ngsim_env/scripts/imitation$ conda install mkl=2017
```

- Seg fault
```bash
signal (11): Segmentation fault
while loading no file, in expression starting on line 0
cd ~/.julia/lib/v0.6
rm PyCall.jl
cd ~/ngsim_env/scripts/imitation
```
- Jupyter kernel dying
```bash
cd ~/.julia/lib/v0.6
rm PyCall.jl
cd ~/ngsim_env/scripts/imitation
```
- No module named 'julia'
```bash
source activate rllab3
cd ~/ngsim_env/python
pip install julia
```
- No module named 'julia_env'
```bash
source activate rllab3
cd ~/rllab
python setup.py develop
 
cd ~/ngsim_env/python
python setup.py develop
 
cd ~/hgail
python setup.py develop
```

These instructions were written by Raunak on 23 March, 2018 to facilitate code delivery to Ashley and Vidya, our technical monitors at Ford.
Edited on 2 April after succesfully running on Ford machines.
