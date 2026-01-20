import argparse

NETWORK_CONFIGS = {
    'final': 'network.network',
}

def get_argparser():
    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of training dataset')
    parser.add_argument('--identifier', type=str, required=True, metavar='N',
                        help='Select the identifier for file name')
    parser.add_argument('--batch-size', type=int,  default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--ch-dim', type=int,  default=64, metavar='N',
                        help='channel dimension for netwrok (default: 64)')
    parser.add_argument('--gradient_accumulation_steps', type=int,  default=1, metavar='N',
                        help='gradient_accumulation_steps for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 100)')
    parser.add_argument('--numlayers', type=int, default=4, metavar='N',
                        help='number of transformer layers(default: 4)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of epoches to log (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--resume', action='store_true',
                        help='resume training by loading last snapshot')
    parser.add_argument('--debug_mode', action='store_true', help='Only load a small subset of data for debugging')
    parser.add_argument('--best_model',type=str, default='best_model',
                        help='path of training snapshot(best model)')
    parser.add_argument('--final_model',type=str, default='final_model',
                        help='path of training snapshot(final model)')
    parser.add_argument('--log_name', type=str, default='init',
                        help='name of the log file')
    parser.add_argument('--network', type=str, default='single',
                        choices=list(NETWORK_CONFIGS.keys()),
                        help=f'Network to use. Choices: {list(NETWORK_CONFIGS.keys())}')
    return parser