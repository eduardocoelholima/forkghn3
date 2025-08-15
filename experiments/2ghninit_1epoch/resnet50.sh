#!/bin/bash

resnet50_state="resnet50-ghn3init-b8e675b-1111"

# echo "This is your conda environment:"
# conda env list|grep "\*"

# move to ghn base dir
cd /home/eclvcs/code/forkghn3

# ghn_init resnet50, trains for 1epoch on cifar10
python train_ddp.py -d cifar10 --arch resnet50 --name resnet50-ghn3init -e 1 --wd 1e-4 -b 128 --lr 0.025 --ckpt /shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-90p-d442e08-1111/checkpoint.pt 

# evaluate trained resnet50
python eval.py -d cifar10 --arch resnet50 --ckpt ./checkpoints/$resnet50_state/checkpoint.pt

# remove resnet checkpoint
# cd /home/eclvcs/code/forkghn3/experiments/2ghninit_1epoch
# ./remove_resnet_checkpoint.sh
