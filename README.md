# Out-of-Distribution (OOD) Detection Benchmarks

Out-of-distribution (OOD) detection is the task of determining whether a datapoint comes from a different distribution than the training dataset. For example, we may train a model to classify the breed of dogs and find that there is a cat image in our dataset. This cat image would be considered out-of-distribution.

OOD detection is useful to find label issues where the actual ground truth label is not in the set of labels for our task (e.g. cat label for a dog breed classification task). This can serve many use-cases, some of which include:

- Remove OOD datapoints from our dataset as part of a data cleaning pipeline
- Consider adding new classes to our task
- Gain deeper insight into the data distribution

This work evaluates the effectiveness of various scores to detect OOD datapoints.

We also present a novel OOD score using the average entropy of K-nearest neighbors.

## Methodology

We treat OOD detection as a binary classification task (True or False: is the datapoint out-of-distribution?) and evaluate the performance of various OOD scores using AUROC.

## Experiments

For each experiment, we perform the following procedure:

1. Train a Neural Network model with ONLY the **in-distribution** training dataset.
2. Use this model to generate predicted probabilties and embeddings for the **in-distribution** and **out-of-distribution** test datasets (these are considered out-of-sample predictions).
3. Use out-of-sample predictions to generate OOD scores.
4. Compute AUROC of OOD scores to detect OOD datapoints.

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

Links below to download the datasets in PNG format for AutoGluon:

**cifar-10** and **cifar-100**
https://github.com/knjcode/cifar2png

**roman-numeral**
https://worksheets.codalab.org/bundles/0x497f5d7096724783aa1eb78b85aa321f

**mnist**
https://github.com/myleott/mnist_png

**fashion-mnist**
https://github.com/DeepLenin/fashion-mnist_png

## Instructions

#### Prerequisite

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker): allows us to properly utilize our NVIDIA GPUs inside docker environments

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

Run the command below to save a pretrained model (ResNet50 trained on ImageNet) in ONNX format. This pretrained model is used in our experiments to compare pre-trained embeddings versus learned embeddings.

```bash
python3 src/image_feature_extraction/convert_feature_extractor_to_onnx.py
```

Run notebook below to train all models.

[src/experiments/OOD/0_Train_Models.ipynb](https://github.com/JohnsonKuan/ood-detection-benchmarks/blob/main/src/experiments/OOD/0_Train_Models.ipynb)

#### 4. Run experiments

Run notebook below to run all experiments.

[src/experiments/OOD/1_Evaluate_All_OOD_Experiments.ipynb](https://github.com/JohnsonKuan/ood-detection-benchmarks/blob/main/src/experiments/OOD/1_Evaluate_All_OOD_Experiments.ipynb)

## Results

Preparation of final results in progress
