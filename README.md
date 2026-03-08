# Customer Next Product Prediction
## Overview

This project builds a next product recommendation system using customer purchase sequences. The goal is to predict the next product a customer is likely to purchase based on previously purchased products.

## Problem Statement

Retail transaction data often contains sequences of purchased items. By learning patterns from these sequences, we can predict the next likely product.

### Applications

* Product recommendation
* Market basket analysis
* Personalized shopping suggestions
* E-commerce recommendation engines

## Models Implemented

### 1. Bigram Model

The Bigram model predicts the next product based on one previous product.

**Formula:** P(product_t | product_{t-1})

It uses frequency counts from the training data to estimate transition probabilities between products.

### 2. Trigram Model

The Trigram model predicts the next product based on two previous products.

**Formula:** P(product_t | product_{t-1}, product_{t-2})

This captures more context but suffers from data sparsity, because many product combinations rarely appear in the dataset.

### 3. Neural Network Model (PyTorch)

A feed-forward neural network is implemented using PyTorch to learn patterns in purchase sequences.

**Architecture:**

* Input (previous products)
* Fully Connected Layer
* ReLU Activation
* Output Layer
* Softmax (probability distribution over products)

**Loss Function:** CrossEntropyLoss

## Evaluation Metrics

### Perplexity

Perplexity measures how well the model predicts a sequence.

**Formula:** Perplexity = exp(CrossEntropyLoss)

### Top-5 Accuracy

Top-5 accuracy measures whether the correct next product appears within the top 5 predicted products.

## Results

### Bigram Model

* Train Perplexity: 25.03
* Validation Perplexity: 25.76
* Train Top-5 Accuracy: 59.30%
* Validation Top-5 Accuracy: 59.54%

### Trigram Model

* Train Perplexity: 12.04
* Validation Perplexity: 6.82
* Train Top-5 Accuracy: 23.04%
* Validation Top-5 Accuracy: 27.43%

### Neural Network Model (PyTorch)

* Train Perplexity: 19.45
* Validation Perplexity: 28.06
* Test Perplexity: 27.96
* Train Top-5 Accuracy: 64.06%
* Validation Top-5 Accuracy: 60.18%
* Test Top-5 Accuracy: 60.34%

## Model Comparison

| Model | Train Perplexity | Validation Perplexity | Validation Top-5 Accuracy |
| --- | --- | --- | --- |
| Bigram | 25.03 | 25.76 | 59.54% |
| Trigram | 12.04 | 6.82 | 27.43% |
| Neural Network | 19.45 | 28.06 | 60.18% |

## Key Observations

* Bigram model performs strongly as a simple baseline
* Trigram suffers from sparsity
* Neural network achieves the best Top-5 accuracy

## Project Structure

* Customer_Sequence/
	+ data/
	+ models/
		- bigram_model.pkl
		- trigram_model.pkl
		- nn_model.pkl
	+ build_bigram.py
	+ build_trigram.py
	+ build_nn.py
	+ README.md

## How to Run

### Install Dependencies

pip install torch numpy pandas

### Train Bigram Model

python build_bigram.py

### Train Trigram Model

python build_trigram.py

### Train Neural Network Model

python build_nn.py

## Key Concepts Demonstrated

* Sequence modeling
* N-gram language models
* Recommender systems
* Data sparsity in probabilistic models
* Neural network prediction using PyTorch

* Evaluation using Perplexity and Top-K Accuracy
