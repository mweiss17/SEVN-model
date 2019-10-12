# SEVN Model
This repository is an extension of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
We

## Installation

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

To get install all the software for the SEVN-model, simply run this script:

```bash

# Create a conda environment for the dependencies (if the sevn environment already exists, don't recreate it)
conda create -n sevn

# Install the SEVN repository and dependencies 
git clone git@github.com:mweiss17/SEVN.git
cd SEVN
pip install -e .
cd ..

# Install the SEVN-model repository and dependencies
git clone git@github.com:mweiss17/SEVN-model.git
cd SEVN-model
pip install -e .
cd ..

# Install the OpenAI Baselines for Atari preprocessing repository and depenencies
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ../SEVN-model
```

Now, let's verify that the install was correct by (briefly) training a model
```bash

python main.py --env-name "SEVN-Train-AllObs-Shaped-v1" \
               --custom-gym SEVN_gym \
               --algo ppo \
               --use-gae \
               --lr 5e-4 \
               --clip-param 0.1 \
               --value-loss-coef 0.5 \
               --num-processes 1 \
               --num-steps 256 \
               --num-mini-batch 4 \
               --log-interval 1 \
               --use-linear-lr-decay \
               --entropy-coef 0.01 \
               --comet mweiss17/navi-corl-2019/UcVgpp0wPaprHG4w8MFVMgq7j \
               --seed 0 \
               --num-env-steps 10000000
```



