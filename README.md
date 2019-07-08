# pytorch-a2c-ppo-acktr

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

pip install tensorflow

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Training example

### Navi

#### PPO

```bash
python main.py --env-name "SEVN-Explorer-v1" --custom-gym SEVN_gym --algo ppo --use-gae --lr 5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --comet mweiss17/navi-corl-2019/UcVgpp0wPaprHG4w8MFVMgq7j --seed 0 --num-env-steps 10000000
```




