import numpy as np
from src.data.data_loader import load_data, split_data
from src.models.neural_network import NeuralNetwork
from src.utilities.metrics import accuracy

def main():
    # Load and preprocess the dataset
    filepath = '/path/to/your/dataset.csv'  # Update this path to your dataset
    data = load_data(filepath)

    # Split the data into training and development sets
    (features_train, labels_train), (features_dev, labels_dev) = split_data(data)

    # Initialize the neural network
    # Note: You need to adjust the input_size, hidden_size, and output_size as per your dataset
    input_size = features_train.shape[0]  # Number of features
    hidden_size = 64  # Number of neurons in the hidden layer
    output_size = 10  # Number of output classes (e.g., 10 for digit recognition)

    neural_net = NeuralNetwork(input_size, hidden_size, output_size)

    # Train the neural network
    iterations = 1000  # Number of iterations for training
    learning_rate = 0.01  # Learning rate
    neural_net.train(features_train, labels_train, iterations, learning_rate)

    # Evaluate the model on the development set
    dev_predictions = neural_net.predict(features_dev)
    dev_accuracy = accuracy(dev_predictions, labels_dev)
    print(f"Development Set Accuracy: {dev_accuracy:.2f}%")

if __name__ == "__main__":
    main()
