# BERT-based Sentiment Analysis with PyTorch

This project leverages BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis on textual data. The implementation uses PyTorch and the Hugging Face Transformers library to train a sequence classification model. The project includes data preparation, model training, evaluation, and prediction on test data.

## Table of Contents

- [Project Overview](#project-overview)
- [Introduction to BERT](#introduction-to-bert)
- [Approach](#approach)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Testing and Prediction](#testing-and-prediction)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project aims to perform sentiment analysis on a dataset provided in TSV (Tab-Separated Values) format. It uses a pre-trained BERT model for sequence classification, fine-tuning it on the sentiment analysis task. The pipeline includes:

1. Data loading and preprocessing.
2. Tokenization using BERT tokenizer.
3. Custom Dataset class for handling the data.
4. Training the BERT model.
5. Evaluating the model on validation data.
6. Predicting sentiment labels for test data.

## Introduction to BERT

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained model developed by Google that has achieved state-of-the-art performance on a wide range of NLP tasks. BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

## Approach

The approach for this project involves the following steps:

1. **Data Loading and Preprocessing**: Load the training and test datasets, and preprocess the text data by tokenizing it using the BERT tokenizer.
2. **Dataset and DataLoader Creation**: Create a custom Dataset class to handle the tokenized text and labels, and use DataLoader to manage batching and shuffling of data.
3. **Model Definition**: Use the pre-trained BERT model for sequence classification, with an additional output layer for binary classification.
4. **Training**: Fine-tune the BERT model on the training data, using AdamW optimizer and cross-entropy loss. The training process involves multiple epochs, with evaluation on a validation set to monitor performance.
5. **Evaluation**: Evaluate the model on the validation set using accuracy and F1 score metrics.
6. **Prediction**: Make predictions on the test data and save the results.

## Dependencies

- Python 3.7+
- pandas
- numpy
- torch
- transformers
- scikit-learn

Install the dependencies using:

```bash
pip install pandas numpy torch transformers scikit-learn
