
# NGSIM Env
- This is a rllab environment for learning human driver models with imitation learning
- This repository does not contain a [gail](https://arxiv.org/abs/1606.03476) / [infogail](https://arxiv.org/abs/1703.08840) / hgail implementation
- It also does not contain the human driver data you need for the environment to work. See [NGSIM.jl](https://github.com/sisl/NGSIM.jl) for that.

## Demo
### GAIL in a sing-agent environment
![gail-multi]()

### Single agent GAIL and PS-GAIL in a multi-agent environment
![gail-multi]()

# Install
- see [`docs/install.md`](docs/install.md)

# Training
- see [`docs/training.md`](docs/training.md)

# Documentation

## How's this work?
- See README files individual directories for details, but a high-level description is:
- The python code uses [pyjulia](https://github.com/JuliaPy/pyjulia) to instantiate a Julia interpreter, see the `python` directory for details
- The driving environment is then built in Julia, see the `julia` directory for details
- Each time the environment is stepped forward, execution passes from python to julia, updating the environment
