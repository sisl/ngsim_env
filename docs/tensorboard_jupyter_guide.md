ssh -L 55555:localhost:55555 dphillips@tver
$> source activate rllab3

Tensor board
On server:
(rllab3)$ cd path/to/experiments/exp_name/imitate/summaries
(rllab3)$ tensorboard --logdir=. --port 55555
On local machine, point browser to:
http://localhost:55556/

Jupyter notebook
On server:
(rllab3)$ cd path/to/ngsim_env/scripts/imitation
(rllab3)$ jupyter notebook --port 55555
[this will produce output like:
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token: ….]
On local machine, point browser to 
    http://localhost:55555/?token=4ffacec7cd35076ab5ae931135a6dd2baefbc35ed0e4ce71

Note:
If trying to do both at the same time, use different ports for them. The simplest way is to have a separate ssh session for each one.

FAQs:
Q: Can I use any port?
A: The requirements you should follow are, 1023 < port number < 65000 (more or less). As long as the port number is not in use this should be fine. I haven’t had any issues with port numbers in use.

Troubleshooting
Tensor board in rllab3 has some weird issues sometimes. Two options:
[1] - create new environment for all future tensor board activity:
$ conda create --name tfenv
$ source activate tfenv
(tfenv)$ pip install tensorboard
[2] - just install tensor board in rllab3 and take the risk it screws something up
(rllab3)$ pip install tensorboard

Jupyter
[1] - you may have to install jupyter. I did:
(rllab3)$ conda install jupyter

