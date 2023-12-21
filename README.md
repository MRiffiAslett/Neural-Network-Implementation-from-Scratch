# Neural Network Implementation from Scratch

## Overview
This project is an implementation of a basic Neural Network from scratch using only NumPy. It's designed to provide a clear understanding of the inner workings of neural networks. The project focuses on the fundamental concepts of neural network operations such as forward propagation, backpropagation, weight updates, and loss calculations.

This implementation is part of my coursework at University College London (UCL) and serves as an educational tool to deepen the understanding of machine learning algorithms.

## Features
- **Pure NumPy Implementation:** No high-level machine learning frameworks are used, ensuring that the core concepts are understood and implemented in detail.
- **Customizable Network Architecture:** Easily adjustable layers, neurons, and activation functions.
- **Basic Neural Network Components:** Includes implementations of forward and backward propagation, cost functions, activation functions, and model evaluation metrics.
- **Dataset Flexibility:** Can be used with any dataset suitable for a neural network.

## Prerequisites
- Python 3.x
- NumPy
- Pandas (for data handling)
- Matplotlib (for visualization, optional)

## Installation
Clone the repository to your local machine:

git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>


Install the necessary dependencies:


## Usage
To use the neural network with your dataset:

1. **Prepare Your Dataset:**
   Ensure your data is in a CSV format with features and labels. The first column should be the labels, and the rest should be features.

2. **Load and Split the Dataset:**
   Update the file path in `examples/train_and_evaluate.py` to point to your dataset.

3. **Train the Model:**
   Run the training script:

