import datetime

import numpy as np
import tqdm
from keras_preprocessing.image import ImageDataGenerator

from src.model.human_model import HumanModel
from src.property.human_property import HumanProperty
from src.property.path_property import PathProperty
from src.repository.human_repository import HumanRepository


class DataAugmentUseCase:
    """
    データオーギュメンテーションのユースケース
    """

    path_property: PathProperty = PathProperty()
    """
    PATHプロパティ
    """

    human_property: HumanProperty = HumanProperty()
    """
    人間プロパティ
    """

    human_repository: HumanRepository = HumanRepository()
    """
    人間リポジトリ
    """

    def handle(self):
        """
        UseCase Handler
        """

        image_data_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            horizontal_flip=True,
            fill_mode='constant',
        )

        humans = self.human_repository.select_train()

        # 各年齢の画像枚数を算出
        age_pictures_size_list: list[int] = []
        for age in range(1, self.human_property.max_age + 1):
            filtered_humans = list(filter(lambda human: human.age == age, humans))
            age_pictures_size_list.append(len(filtered_humans))

        # 各年齢のデータを水増し
        # TODO: 全部保存するのではなく、学習に使う分だけ保存する（現在は手動で仕分けている）
        augmentation_rate: int = 8
        for age in tqdm.tqdm(range(1, self.human_property.max_age + 1)):
            filtered_humans = list(filter(lambda x: x.age == age, humans))
            if len(filtered_humans) == 0:
                continue

            x: np.ndarray = np.array([human.image for human in filtered_humans])
            y: np.ndarray = np.array([[human.age, human.gender, human.race] for human in filtered_humans])

            count: int = 0
            for data in image_data_generator.flow(x, y, batch_size=1):
                number_of_pictures: int = age_pictures_size_list[age - 1] + count
                if number_of_pictures >= age_pictures_size_list[age - 1] * augmentation_rate:
                    break

                gender = int(data[1][0][1])
                race = int(data[1][0][2])
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                human = HumanModel(
                    age=age,
                    gender=gender,
                    race=race,
                    image=np.array(data[0][0]),
                    filename=f"{age}_{gender}_{race}_{timestamp}.jpg.chip.jpg",
                )
                self.human_repository.save(human, True)

                count += 1
