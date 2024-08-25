# README

## Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) model using the MNIST dataset. The CNN model is built using Keras and TensorFlow frameworks, and a custom activation function named **Parametric Rectified Sigmoid Exponential Linear Unit (PPRsigELU)** is introduced and used. The goal of the project is to classify handwritten digits from the MNIST dataset using this custom activation function to explore its impact on model performance.

## Project Structure

- **Data Loading and Preprocessing:**
  - The MNIST dataset is loaded and split into training and test sets.
  - Data is normalized and reshaped to fit the input requirements of the CNN.

- **Custom Activation Function:**
  - A new activation function called **PPRsigELU** is defined and used in the CNN layers.
  - The activation function is designed to provide better learning capabilities by introducing two learnable parameters, `alpha` and `beta`.

- **Model Creation:**
  - A Sequential CNN model is created with convolutional layers, batch normalization, max-pooling, and dropout for regularization.
  - The custom activation function is applied after each convolutional layer.

- **Model Training:**
  - The model is compiled using Stochastic Gradient Descent (SGD) as the optimizer.
  - The model is trained on the MNIST dataset and the training loss and accuracy are plotted.

- **Model Evaluation:**
  - The trained model is evaluated on both the training and test sets to check for overfitting and generalization.

- **Model Saving:**
  - The trained model is saved for future inference or further training.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib


## How to Run

1. **Load the Data:**
   The MNIST dataset is automatically downloaded and loaded using `mnist.load_data()`.

2. **Preprocess the Data:**
   The images are normalized to the range [0, 1] and reshaped to fit the input shape of the CNN.

3. **Create the Model:**
   The CNN model is created using the `create_cnn()` function, which includes convolutional layers with the custom activation function.

4. **Train the Model:**
   The model is trained using the `model.fit()` function, where you can specify the number of epochs and batch size.

5. **Evaluate the Model:**
   The model's performance is evaluated on the test set, and the accuracy is printed.

6. **Save the Model:**
   The trained model is saved as an HDF5 file for future use.

## The Custom Activation Function (PPRsigELU)

### Mathematical Expression

The **Parametric Rectified Sigmoid Exponential Linear Unit (PPRsigELU)** activation function is defined as:

$$
f(x) =
\begin{cases}
x \cdot \sigma(x) \cdot \alpha + x & \text{if } x > 1.0 \\
x & \text{if } 0 \leq x \leq 1.0 \\
\beta \cdot (\exp(x) - 1) & \text{if } x < 0
\end{cases}
$$

Where:
- $$\ \sigma(x) = \frac{1}{1 + \exp(-x)} $$ is the sigmoid function.
- $$\ \alpha $$ and $$\ \beta $$ are learnable parameters initialized using the `GlorotNormal` initializer.

## Model Performance

- **Training Accuracy and Loss:**
  - The accuracy and loss during training are plotted and analyzed to understand the model's learning behavior.
  - $$\ Train Accuracy: 99.90$$%
  - $$\ Train Loss:  0.00352$$


- **Test Evaluation:**
  - The final accuracy on the test set is computed to evaluate the model's generalization capability.
  - $$\ Test Accuracy: 98.89$$%
  - $$\ Test Loss:  0.04715$$
## Conclusion

This project explores the implementation of a custom activation function in a CNN model for digit classification using the MNIST dataset. The PPRsigELU activation function introduces flexibility through learnable parameters, potentially improving the model's performance. Further experiments can be conducted to compare this activation function with other standard functions like ReLU, ELU, and Sigmoid to analyze its effectiveness.
