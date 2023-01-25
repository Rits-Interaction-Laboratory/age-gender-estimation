import argparse

from src.nnet.resnet import ResNet
from src.usecase.analyse_usecase import AnalyseUseCase
from src.usecase.data_augment_usecase import DataAugmentUseCase
from src.usecase.estimate_usecase import EstimateUseCase
from src.usecase.train_usecase import TrainUseCase

# アプリケーションのオプションを定義
# --helpでヘルプを表示できる
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser()
argument_parser.add_argument("-t", "--train",
                             help="学習",
                             action="store_true")
argument_parser.add_argument("-e", "--estimate",
                             help="年齢推定",
                             action="store_true")
argument_parser.add_argument("-l", "--log",
                             help="ログ分析",
                             action="store_true")
argument_parser.add_argument("-d", "--data_augment",
                             help="データオーギュメンテーション",
                             action="store_true")
argument_parser.add_argument("-w", "--weights",
                             help="学習済みモデルのファイル名",
                             type=str)
argument_parser.add_argument("-f", "--filename",
                             help="ファイル名",
                             type=str)
arguments = argument_parser.parse_args()

if arguments.train:
    resnet = ResNet()
    train_usecase = TrainUseCase(resnet)
    history = train_usecase.handle()

    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    print(f"最良エポック：{best_epoch}")
    weights_filename = (f"ckpt/{best_epoch}.h5")

    estimate_usecase = EstimateUseCase(resnet, weights_filename)
    estimate_usecase.handle()
elif arguments.estimate:
    estimate_usecase = EstimateUseCase(ResNet(), arguments.weights)
    estimate_usecase.handle()
elif arguments.log:
    analyse_usecase = AnalyseUseCase()
    analyse_usecase.handle(arguments.filename)
elif arguments.data_augment:
    data_augment_usecase = DataAugmentUseCase()
    data_augment_usecase.handle()
else:
    argument_parser.print_help()
