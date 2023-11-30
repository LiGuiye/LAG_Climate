import argparse

import utils.utils as tool
from trainers.lag_trainer import LAGTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAG_Climate')

    # dataset
    parser.add_argument('--dataset_name', type=str, default='Solar_07-14',help='Dataset name: Wind_07-14 or Solar_07-14.')
    parser.add_argument('--channel', type=int, default=2, help='Image channels')
    parser.add_argument('--data_size', type=int, default=512, help='Nearest Neighbor interpolation will be used to resize if actual image size is different.')

    # nets
    parser.add_argument('--noise_dim', type=int, default=64)
    parser.add_argument('--residual_scaling', type=float, default=0.125)
    parser.add_argument('--residual_blocks', type=int, default=8, help='Number of residual blocks in EDSR.')
    parser.add_argument('--max_scale', type=int, default=64, help='The maximum SR scale you want to train')
    parser.add_argument('--n_critic', type=int, default=4, help='Train D how many times while train G 1 time.')
    parser.add_argument('--kernelSize', type=int, default=1, help='Kernel size of conv in to_rgb and from_rgb')
    parser.add_argument('--mse_weight', type=int, default=10, help='Weight of L_center in loss function for G.')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--report_step', type=int, default=100)
    parser.add_argument('--trial_name', type=str, default="debug")
    parser.add_argument('--memtest', type=bool, default=False, help='start from the largest scale factor')
    parser.add_argument('--reset', type=tool.boolean_string, default=True, help='Retrain from the start.')
    parser.add_argument('--epoch', type=str, default='2,2', help='epochs for transition and stabilization stages')

    args = parser.parse_args()

    tool.set_random_seed(666)
    args.sample_indices = [0]

    trainer = LAGTrainer(args)
    trainer.train_epoch()
