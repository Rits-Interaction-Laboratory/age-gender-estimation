from keras import Sequential, layers

from src.nnet.base_nnet import BaseNNet


class CNN(BaseNNet):

    def build_model(self):
        """
        モデルをビルド
        """

        self.model = Sequential()

        self.model.add(layers.Input(shape=self.human_property.shape))

        # convolution 1st layer
        self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D())

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.BatchNormalization())

        # fully connected final layer
        self.model.add(layers.Dense(2, activation="sigmoid"))
