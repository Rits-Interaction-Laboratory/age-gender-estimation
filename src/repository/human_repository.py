import glob
import os

import numpy as np
from keras_preprocessing.image import img_to_array, load_img

from src.model.human_model import HumanModel
from src.property.path_property import PathProperty


class HumanRepository:
    """
    人間リポジトリ
    """

    path_property: PathProperty = PathProperty()
    """
    PATHプロパティ
    """

    def select_all(self) -> list[HumanModel]:
        """
        人間モデルを全件取得

        :return: 人間モデルリスト
        """

        filenames: list[str] = glob.glob(f"{self.path_property.data_path}/*.jpg")
        return list(map(
            lambda filename: self.select_by_filename(filename),
            filenames
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

    def split_dataset(self, humans: list[HumanModel], validation_split_rate: float) -> list[list[HumanModel]]:
        """
        データセットを学習用、検証用に分割

        :param humans: 人間リスト
        :param validation_split_rate: 検証用データの割合
        :return: [学習用, 検証用]
        """

        np.random.shuffle(humans)
        split_index: int = int(validation_split_rate * len(humans))
        humans_train = humans[split_index:]
        humans_test = humans_train[0:split_index]

        return [humans_train, humans_test]
