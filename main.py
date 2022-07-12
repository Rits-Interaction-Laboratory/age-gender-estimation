import argparse

from src.nnet.cnn import CNN
from src.usecase.train_usecase import TrainUseCase

# アプリケーションのオプションを定義
# --helpでヘルプを表示できる
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser()
argument_parser.add_argument('-t', '--train',
                             help='学習',
                             action='store_true')
arguments = argument_parser.parse_args()

if arguments.train:
    train_usecase = TrainUseCase(CNN())
    train_usecase.train()
else:
    argument_parser.print_help()
