import csv
import os.path

from matplotlib import pyplot as plt

from src.property.path_property import PathProperty


class AnalyseUseCase:
    """
    分析のユースケース
    """

    path_property: PathProperty = PathProperty()
    """
    PATHプロパティ
    """

    def analyse_log(self, filename: str):
        """
        学習ログを分析
        """

        loss_list: list[float] = []
        val_loss_list: list[float] = []
        θ_metric_list: list[float] = []
        val_θ_metric_list: list[float] = []
        σ_metric_list: list[float] = []
        val_σ_metric_list: list[float] = []

        with open(filename, "r") as f:
            # ヘッダーは読み飛ばす
            next(csv.reader(f))

            for row in csv.reader(f):
                columns = [float(column) for column in row]
                epoch, loss, val_loss, val_θ_metric, val_σ_metric, θ_metric, σ_metric = columns

                loss_list.append(loss)
                val_loss_list.append(val_loss)
                θ_metric_list.append(θ_metric)
                val_θ_metric_list.append(val_θ_metric)
                σ_metric_list.append(σ_metric)
                val_σ_metric_list.append(val_σ_metric)

        # グラフを保存するディレクトリを作成
        dirname = f"{self.path_property.log_path}/{os.path.splitext(os.path.basename(filename))[0]}"
        os.makedirs(dirname, exist_ok=True)

        # lossのグラフを出力
        plt.figure()
        plt.plot(range(len(loss_list)), loss_list, label="train")
        plt.plot(range(len(loss_list)), val_loss_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(f"{dirname}/loss.png")

        # θのグラフを出力
        plt.figure()
        plt.plot(range(len(loss_list)), θ_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_θ_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("θ mae")
        plt.savefig(f"{dirname}/θ.png")

        # σのグラフを出力
        plt.figure()
        plt.plot(range(len(loss_list)), σ_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_σ_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("σ mae")
        plt.savefig(f"{dirname}/σ.png")
