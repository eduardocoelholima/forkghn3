#!/bin/bash

dataset="cifar10"
arch="googlenet"
name="tmp"
gitid=`git log -1 | grep ^commit | cut -d " " -f 2|head -c 7`
seed="1111"
nn_ckpt="./checkpoints/$name-$gitid-$seed/checkpoint.pt"
cd /home/eclvcs/code/forkghn3 # move to ghn base dir

ckpt1="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-1p-c45a103-1111/checkpoint.pt"
ckpt2="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-2p-c45a103-1111/checkpoint.pt"
ckpt3="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-3p-c45a103-1111/checkpoint.pt"
ckpt4="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-4p-c45a103-1111/checkpoint.pt"
ckpt5="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-5p-c45a103-1111/checkpoint.pt"
ckpt6="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-6p-c45a103-1111/checkpoint.pt"
ckpt7="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-7p-c45a103-1111/checkpoint.pt"
ckpt8="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-8p-c45a103-1111/checkpoint.pt"
ckpt9="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-9p-c45a103-1111/checkpoint.pt"
ckpt10="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-10p-b3289ac-1111/checkpoint.pt"
ckpt11="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-20p-b3289ac-1111/checkpoint.pt"
ckpt12="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-30p-b3289ac-1111/checkpoint.pt"
ckpt13="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-40p-b3289ac-1111/checkpoint.pt"
ckpt14="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-50p-b3289ac-1111/checkpoint.pt"
ckpt15="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-60p-d442e08-1111/checkpoint.pt"
ckpt16="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-70p-d442e08-1111/checkpoint.pt"
ckpt17="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-80p-d442e08-1111/checkpoint.pt"
ckpt18="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-90p-d442e08-1111/checkpoint.pt"
ckpt19="/shared/rc/nlagent/checkpoints/cifar10/ghn3tm8-c10-b3289ac-1111/checkpoint.pt"

for i in $(seq 1 19); do
    declare -n ghn_ckpt=ckpt$i
    echo $ghn_cpkt
    python train_ddp.py -d $dataset --arch $arch --name $name -e 1 --wd 1e-4 -b 128 --lr 0.025 \
        --ckpt $ghn_ckpt #ghn_init, train 1epoch
    python eval.py -d $dataset --arch $arch --ckpt $nn_ckpt # evaluate trained nn
    rm -rf $nn_ckpt # remove nn checkpoint
done

# ghn3tm8-c10-1p-c45a103-1111
# ghn3tm8-c10-2p-c45a103-1111
# ghn3tm8-c10-3p-c45a103-1111
# ghn3tm8-c10-4p-c45a103-1111
# ghn3tm8-c10-5p-c45a103-1111
# ghn3tm8-c10-6p-c45a103-1111
# ghn3tm8-c10-7p-c45a103-1111
# ghn3tm8-c10-8p-c45a103-1111
# ghn3tm8-c10-9p-c45a103-1111
# ghn3tm8-c10-10p-b3289ac-1111

# ghn3tm8-c10-20p-b3289ac-1111
# ghn3tm8-c10-30p-b3289ac-1111
# ghn3tm8-c10-40p-b3289ac-1111
# ghn3tm8-c10-50p-b3289ac-1111
# ghn3tm8-c10-60p-d442e08-1111
# ghn3tm8-c10-70p-d442e08-1111
# ghn3tm8-c10-80p-d442e08-1111  
# ghn3tm8-c10-90p-d442e08-1111
# ghn3tm8-c10-b3289ac-1111

# ghn3tm8-c10-90p1-08084b1-1111
# ghn3tm8-c10-90p1-5996d33-1111
# ghn3tm8-c10-90p1-63ccb05-1111
# ghn3tm8-c10-90p1-d442e08-1111
# ghn3tm8-c10-5c-948769b-1111
# ghn3tm8-c10-7c-948769b-1111
# ghn3tm8-c10-50p-5c-d442e08-1111
# ghn3tm8-c10-60p-5c-d442e08-1111
# ghn3tm8-c10-70p-5c-d442e08-1111
# ghn3tm8-c10-5labels-3c8df8c-1111
# ghn3tm8-c10custom-2867d32-1111

# echo "This is your conda environment:"
# conda env list|grep "\*" # shows current conda env
# cd /home/eclvcs/code/forkghn3/experiments/2ghninit_1epoch # comes back
