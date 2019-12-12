pip install --user -e /home/dockeruser/SEVN
python /home/dockeruser/SEVN-model/main.py --env-name "SEVN-Train-AllObs-Shaped-0-v1" \
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
--num-env-steps 1000000