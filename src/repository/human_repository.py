import glob
import os

import numpy as np
import tqdm
from keras_preprocessing.image import img_to_array, load_img

from src.model.human_model import HumanModel
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

        np.random.shuffle(humans)
        split_index: int = int(self.nnet_property.validation_split_rate * len(humans))
        humans_train = humans[split_index:]
        humans_test = humans_train[0:split_index]

        return [humans_train, humans_test]
