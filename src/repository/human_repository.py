import glob
import json
import os

import numpy as np
import tqdm
from keras_preprocessing.image import img_to_array, load_img

from src.model.human_model import HumanModel
from src.property.logging_property import LoggingProperty
from src.property.nnet_property import NNetProperty
from src.property.path_property import PathProperty


class HumanRepository:
    """
    人間リポジトリ
    """

    path_property: PathProperty = PathProperty()
    """
    PATHプロパティ
    """

    nnet_property: NNetProperty = NNetProperty()
    """
    ニューラルネットプロパティ
    """

    logging_property: LoggingProperty = LoggingProperty()
    """
    ロギングプロパティ
    """

    def select_all(self) -> list[HumanModel]:
        """
        人間モデルを全件取得

        :return: 人間モデルリスト
        """

        filenames: list[str] = glob.glob(f"{self.path_property.data_path}/*.jpg")
        filenames = filenames[:int(self.nnet_property.usage_rate * len(filenames))]
        return list(map(
            lambda filename: self.select_by_filename(filename),
            tqdm.tqdm(filenames)
        ))

    def select_by_filename(self, filename: str) -> HumanModel:
        """
        ファイル名から人間を取得

        :param filename: ファイル名
        :return: 人間モデル
        """

        # ファイル名の命名規則は下記を参照
        # https://susanqq.github.io/UTKFace/
        age, gender, race, _ = os.path.basename(filename).split("_")

        # 整形済みのコーパスを利用しているので、前処理は不要
        image: np.ndarray = img_to_array(load_img(filename))

        return HumanModel(
            age=int(age),
            gender=int(gender),
            race=int(race),
            image=image,
            filename=filename
        )

    def split_dataset(self, humans: list[HumanModel]) -> list[list[HumanModel]]:
        """
        データセットを学習用、検証用に分割

        :param humans: 人間リスト
        :return: [学習用, 検証用]
        """

        dataset_loader_log: dict = self.__get_dataset_loader_log(humans)
        humans_train = list(filter(lambda human: human.filename in dataset_loader_log["train_filenames"], humans))
        humans_test = list(filter(lambda human: human.filename in dataset_loader_log["test_filenames"], humans))

        return [humans_train, humans_test]

    def __get_dataset_loader_log(self, humans: list[HumanModel]) -> dict:
        """
        データセット読み込みのログを取得（存在しない場合は作成）

        :param humans: 人間リスト
        :return: ログ
        """

        # 前回のログが存在する場合は再利用する
        dataset_loader_log_filename = f"{self.path_property.log_path}/{self.logging_property.dataset_loader_filename}"
        if os.path.exists(dataset_loader_log_filename):
            with open(dataset_loader_log_filename, "r") as f:
                dataset_loader_log: dict = json.load(f)

                # 前回から検証用データの割合が変更されていない場合はreturnする
                if self.nnet_property.validation_split_rate == dataset_loader_log["validation_split_rate"]:
                    return dataset_loader_log

        # データセットを学習用、検証用に分割
        np.random.shuffle(humans)
        split_index: int = int(self.nnet_property.validation_split_rate * len(humans))
        humans_train = humans[split_index:]
        humans_test = humans[0:split_index]

        # ログ作成
        dataset_loader_log = {
            "train_filenames": [human.filename for human in humans_train],
            "test_filenames": [human.filename for human in humans_test],
            "validation_split_rate": self.nnet_property.validation_split_rate,
        }

        # ログ出力
        dataset_loader_log_filename = f"{self.path_property.log_path}/{self.logging_property.dataset_loader_filename}"
        with open(dataset_loader_log_filename, "w") as f:
            json.dump(dataset_loader_log, f, indent=2)

        return dataset_loader_log
