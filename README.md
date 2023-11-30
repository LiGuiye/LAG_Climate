# LAG_Climate

SR results (up to 64 $\times$ ) for the example climate datasets.

**Solar irradiance (DNI & DHI)**

![](results/Solar/Solar_07-14_bs1_epoch15_lr4e-3_64X/eval/test_samples_[1294]_different_scales.png)

**Wind velocity (U & V)**

![](results/Wind/Wind_07-14_bs16_epoch30_lr2e-3_64X/eval/test_samples_[1294]_different_scales.png)

----

Table of Contents
- [Requirements](#requirements)
- [Pre-trained weights](#pre-trained-weights)
- [Model training](#model-training)
- [Model testing](#model-testing)

## Requirements

Here is the list of libraries you need to install to execute the code:

- python = 3.6.13
- pytorch = 1.10.2
- scipy
- numpy
- matplotlib
- torchvision

## Pre-trained weights

Pre-trained weights and corresponding model configures can be found in `Results/**/**/args` and `Results/**/**/checkpoint`. Checkpoints of models at different SR scales can be downloaded from [OneDrive](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/guli2564_colorado_edu/ErmiDRBV8YtCrF-n6EEcFGwBJgS4XIAMwnOtt6GrWqasdg?e=oWmguT).

## Model training

```shell
python train.py --dataset_name Solar_07-14 --lr 0.004 --batch_size 1 --epoch 15,15 --report_step 140000 --trial_name Solar/Solar_07-14_bs1_epoch15_lr4e-3_64X --reset True

python train.py --dataset_name Wind_07-14 --lr 0.002 --batch_size 16 --epoch 30,30 --report_step 7000 --trial_name Wind/Wind_07-14_bs16_epoch30_lr2e-3_64X --reset True
```

## Model testing

```shell
python test.py
```

