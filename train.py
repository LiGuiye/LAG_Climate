import argparse

from utils.utils import set_random_seed, boolean_string
from trainers.lag_trainer import LAGTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAG_Climate')

    # dataset
    parser.add_argument('--data_name', type=str, default='Wind',help='Dataset name: Wind or Solar.')
    parser.add_argument('--channel', type=int, default=2, help='Image channels')
    parser.add_argument('--data_size', type=int, default=512, help='Nearest Neighbor interpolation will be used for resizing.')

    # nets
    parser.add_argument('--noise_dim', type=int, default=64)
    parser.add_argument('--residual_scaling', type=float, default=0.125)
    parser.add_argument('--residual_blocks', type=int, default=8, help='Num of EDSR residual blocks in G and D')
    parser.add_argument('--max_scale', type=int, default=64, help='The maximum SR scale you want to train')
    parser.add_argument('--n_critic', type=int, default=4, help='Train D how many times while train G 1 time.')
    parser.add_argument('--kernelSize', type=int, default=1, help='Kernel size of conv in toImg and fromImg.')
    parser.add_argument('--vario_weight', type=int, default=0, help='Weight of L_vario in loss function for G.')
    parser.add_argument('--mass_weight', type=int, default=0, help='Weight of L_mass in loss function for G.')
    parser.add_argument('--center_weight', type=int, default=10, help='Weight of L_center in loss function for G.')

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--report_step', type=int, default=10)
    parser.add_argument('--expid', type=str, default="debug")
    parser.add_argument('--memtest', type=bool, default=False, help='start from the largest scale factor')
    parser.add_argument('--reset', type=boolean_string, default=True, help='Retrain from the start.')
    parser.add_argument('--epoch', type=str, default='30,30', help='epochs for transition and stabilization stages')
    args = parser.parse_args()

    set_random_seed()
    trainer = LAGTrainer(args)
    trainer.train_epoch()