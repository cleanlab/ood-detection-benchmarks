# Out-of-Distribution (OOD) Detection Benchmarks

Code to reproduce results from the paper:

[**Back to the Basics: Revisiting Out-of-Distribution Detection Baselines**](https://arxiv.org/abs/2207.03061). [*ICML 2022 Workshop on Principles of Distribution Shift*](https://icml.cc/Conferences/2022/ScheduleMultitrack?event=20541)

Out-of-distribution (OOD) detection is the task of determining whether a datapoint comes from a different distribution than the training dataset. For example, we may train a model to classify the breed of dogs and find that there is a cat image in our dataset. This cat image would be considered out-of-distribution. 
This work evaluates the effectiveness of various scores to detect OOD datapoints.

This repository is only for intended for scientific purposes. To detect outliers in your own data, you should instead use the [implementation](https://docs.cleanlab.ai/stable/tutorials/outliers.html) from the official [cleanlab](https://github.com/cleanlab/cleanlab) library.

## File Structure
This repository is broken into two major folders (inside `src/experiments/`):

1. `OOD/`: primary benchmarking code used for the paper linked above.

2. `adjusted-OOD-scores/`: additional benchmarking code to produce results from the article:

[A Simple Adjustment Improves Out-of-Distribution Detection for Any Classifier](https://pub.towardsai.net/a-simple-adjustment-improves-out-of-distribution-detection-for-any-classifier-5e96bbb2d627). *Towards AI*, 2022.

This additional code considers OOD detection based solely on classifier predictions and adjusted versions thereof. 


## Experiments

For each experiment, we perform the following procedure:

1. Train a Neural Network model with ONLY the **in-distribution** training dataset.
2. Use this model to generate predicted probabilties and embeddings for the **in-distribution** and **out-of-distribution** test datasets (these are considered out-of-sample predictions).
3. Use out-of-sample predictions to generate OOD scores.
4. Threshold OOD scores to detect OOD datapoints.

| Experiment ID | In-Distribution | Out-of-Distribution |
| :------------ | :-------------- | :------------------ |
| 0             | cifar-10        | cifar-100           |
| 1             | cifar-100       | cifar-10            |
| 2             | mnist           | roman-numeral       |
| 3             | roman-numeral   | mnist               |
| 4             | mnist           | fashion-mnist       |
| 5             | fashion-mnist   | mnist               |



## Download datasets

For our experiments, we use [AutoGluon's ImagePredictor](https://auto.gluon.ai/dev/tutorials/image_prediction/beginner.html) for image classification which requires the training, validation, and test datasets to be image files.

Links below to download the training and test datasets in PNG format:

- **cifar-10** and **cifar-100**:
  https://github.com/knjcode/cifar2png

- **roman-numeral**:
  https://worksheets.codalab.org/bundles/0x497f5d7096724783aa1eb78b85aa321f

  There are duplicate images in the dataset (exact same image with different file names). We use the following script to dedupe: `src/preprocess/remove_dupes.py`

- **mnist**:
  https://github.com/myleott/mnist_png

- **fashion-mnist**:
  https://github.com/DeepLenin/fashion-mnist_png


## Instructions to reproduce results

#### Prerequisite

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker): allows us to properly utilize our NVIDIA GPUs inside docker environments
- [autogluon](https://auto.gluon.ai/stable/index.html)==0.4.0


#### 1. Run docker-compose to build the docker image and run the container

Clone this repo and run below commands:

```bash
sudo docker-compose build
sudo docker-compose run --rm --service-port dcai
```


#### 2. Start Jupyter Lab

Run command below.

Note that we use a Makefile to run jupyter lab for convenience so we can save args (ip, port, allow-root, etc).

```bash
make jupyter-lab
```


#### 3. Train models

Run notebook below to train all models.

[src/experiments/OOD/0_Train_Models.ipynb](https://github.com/JohnsonKuan/ood-detection-benchmarks/blob/main/src/experiments/OOD/0_Train_Models.ipynb)

Note that we use 2 neural net architectures below with AutoGluon and each use different backends:

- swin_base_patch4_window7_224 (torch backend)
- resnet50_v1 (mxnet backend)


#### 4. Run experiments

Here is a notebook that runs all experiments:

[src/experiments/OOD/1_Evaluate_All_OOD_Experiments.ipynb](https://github.com/JohnsonKuan/ood-detection-benchmarks/blob/main/src/experiments/OOD/1_Evaluate_All_OOD_Experiments.ipynb)
