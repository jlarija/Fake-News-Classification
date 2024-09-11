# Fake-News-Classification

## Overview

This repository contains a machine learning classification project for detecting fake news and topic detection. The project utilizes a feedforward neural network with no hidden layers and compares its performance with a classic tree classification method, xgboost. Parameter tuning is also conducted.

## Project Description

The goal of this project is to classify news articles as either real or fake using a simple feedforward neural network model. Given the dataset's limited size and complexity, the model is kept minimal by omitting hidden layers. This approach is evaluated against several classification methods to determine its effectiveness.

## Dataset

The dataset used for this project is comprised of news articles labeled as fake or real. The dataset was obtained on Kaggle.

## Model
s
### Neural Network

- **Architecture**: Feedforward neural network with no hidden layers.
- **Activation Function**: Sigmoid (binary classification problem)
- **Loss Function**: The BCELoss (Binary Cross Entropy Loss) measures the difference between the predicted probability and the actual binary label (0 or 1)
- **Optimizer**: Adam


## Evaluation

The performance of the neural network model and the classification methods is compared based on metrics such as accuracy, precision, recall, and F1-score. In the EVAL notebook, relevant graphs and model evaluation is provided. 

## Usage

1. **Run Word2Vec vectorization for the data**
```bash
python main.py word2vec '../input/df_news.csv' -path_out '../input/df_news.pkl'
```
2. **Run feedforward neural network in pytorch for classification**
```bash
python main.py train_nn '../input/df_news.pkl' -path_out '../models/forward_nn.pth'
```
3. **Run xgboost classifier**
```bash
python main.py train_xgboost '../input/df_news.pkl' -path_out '../models/forward_nn.pth'
```