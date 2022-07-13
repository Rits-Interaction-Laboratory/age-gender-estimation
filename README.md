# Age Estimation

![CI](https://github.com/Rits-Interaction-Laboratory/age-gender-estimation/workflows/CI/badge.svg)
![python version](https://img.shields.io/badge/python_version-3.9-blue.svg)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Keras implementation of a CNN network for age estimation.

## Requirement

- Python 3.9
- pipenv

## Development

### Installation

```bash
$ pipenv install --dev
```

### Code Quality

```bash
# code check
$ pipenv run lint && pipenv run mypy

# code format
$ pipenv run format
```

## Usage

### Preparation

You need to download [UTKFace](https://susanqq.github.io/UTKFace/) and place the data in `data` directory.

And then, you need to copy `resources/application-sample.yml` to `resources/application.yml`.
This file is application preferences, so please rewrite it if necessary.

### Run Application

You can run this application from pipenv.

```bash
$ pipenv run start --train
```

Please refer to the help for more detailed usage.

```
$ pipenv run start --help
usage: main.py [-h] [-r] [-c]

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           学習
  -l, --log             ログ分析
```

### Run on docker

After downloading the dataset, execute the following command.

```bash
# build docker image
$ docker build -t age-gender-estimation .

# train
$ docker run --gpus=all \
  -v $HOME/age-gender-estimation/resources:/app/resources \
  -v $HOME/age-gender-estimation/log:/app/log \
  -it age-gender-estimation python main.py --train

# estimate & export heatmap
$ docker run --gpus=all \
  -v $HOME/age-gender-estimation/resources:/app/resources \
  -v $HOME/age-gender-estimation/log:/app/log \
  -it age-gender-estimation python main.py --estimation
```
