# Customer Next Product Prediction

Overview

This project builds a next product recommendation system using customer purchase sequences. The goal is to predict the next product a customer is likely to purchase based on previously purchased products.

The problem is treated similarly to language modeling, where products are tokens and purchase histories are sequences.

Three different approaches are implemented and compared:

Bigram Model

Trigram Model

Neural Network Model (PyTorch)

Each model is evaluated using Perplexity and Top-5 Accuracy, which are common metrics in sequence prediction and recommender systems.

Problem Statement

Retail transaction data often contains sequences of purchased items. By learning patterns from these sequences, we can predict the next likely product.

Applications include:

Product recommendation

Market basket analysis

Personalized shopping suggestions

E-commerce recommendation engines

Example purchase sequence:

[256, 241, 85123A, 84879]

Meaning a customer bought product 256, then 241, then 85123A, etc.

The goal is to predict the next product in the sequence.

Models Implemented
1. Bigram Model

The Bigram model predicts the next product based on one previous product.

Formula:

P(product_t | product_{t-1})

It uses frequency counts from the training data to estimate transition probabilities between products.

2. Trigram Model

The Trigram model predicts the next product based on two previous products.

Formula:

P(product_t | product_{t-1}, product_{t-2})

This captures more context but suffers from data sparsity, because many product combinations rarely appear in the dataset.

3. Neural Network Model (PyTorch)

A feed-forward neural network is implemented using PyTorch to learn patterns in purchase sequences.

Architecture:

Input (previous products)
        ↓
Fully Connected Layer
        ↓
ReLU Activation
        ↓
Output Layer
        ↓
Softmax (probability distribution over products)

Loss Function:

CrossEntropyLoss
Evaluation Metrics
Perplexity

Perplexity measures how well the model predicts a sequence.

Lower perplexity indicates better predictive performance.

Formula:

Perplexity = exp(CrossEntropyLoss)
Top-5 Accuracy

Top-5 accuracy measures whether the correct next product appears within the top 5 predicted products.

Example:

Predicted products:

[85123A, 22423, 85099B, 47566, 20725]

If the actual product is in this list, the prediction is considered correct.

Results
Bigram Model

Train Perplexity: 25.03
Validation Perplexity: 25.76

Train Top-5 Accuracy: 59.30%
Validation Top-5 Accuracy: 59.54%

Example prediction after product 256:

85123A (0.4819)

84879 (0.0154)

POST (0.0069)

22961 (0.0067)

22138 (0.0067)

Trigram Model

Train Perplexity: 12.04
Validation Perplexity: 6.82

Train Top-5 Accuracy: 23.04%
Validation Top-5 Accuracy: 27.43%

Example prediction after products 256 and 241:

85123A (0.5943)

22423 (0.2532)

85099B (0.2532)

47566 (0.2526)

20725 (0.2526)

Note: Trigram models suffer from data sparsity, which reduces Top-K accuracy despite lower perplexity.

Neural Network Model (PyTorch)

Train Perplexity: 19.45
Validation Perplexity: 28.06
Test Perplexity: 27.96

Train Top-5 Accuracy: 64.06%
Validation Top-5 Accuracy: 60.18%
Test Top-5 Accuracy: 60.34%

Example prediction:

85123A (0.6799)

22079 (0.0349)

22077 (0.0162)

22457 (0.0059)

20972 (0.0053)

Model Comparison
Model	Train Perplexity	Validation Perplexity	Validation Top-5 Accuracy
Bigram	25.03	25.76	59.54%
Trigram	12.04	6.82	27.43%
Neural Network	19.45	28.06	60.18%

Key observations:

Bigram model performs strongly as a simple baseline

Trigram suffers from sparsity

Neural network achieves the best Top-5 accuracy

Project Structure
Customer_Sequence/
│
├── data/
│
├── models/
│   ├── bigram_model.pkl
│   ├── trigram_model.pkl
│   └── nn_model.pkl
│
├── build_bigram.py
├── build_trigram.py
├── build_nn.py
│
└── README.md
How to Run
Install Dependencies
pip install torch numpy pandas
Train Bigram Model
python build_bigram.py
Train Trigram Model
python build_trigram.py
Train Neural Network Model
python build_nn.py
Key Concepts Demonstrated

This project demonstrates:

Sequence modeling

N-gram language models

Recommender systems

Data sparsity in probabilistic models

Neural network prediction using PyTorch

Evaluation using Perplexity and Top-K Accuracy