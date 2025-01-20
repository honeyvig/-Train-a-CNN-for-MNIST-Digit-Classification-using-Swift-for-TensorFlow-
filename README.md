# -Train-a-CNN-for-MNIST-Digit-Classification-using-Swift-for-TensorFlow
Training a Convolutional Neural Network (CNN) for MNIST digit classification using Swift for TensorFlow is a great way to learn how to apply deep learning techniques in Swift. Swift for TensorFlow enables easy and powerful machine learning workflows with a native Swift API, and it integrates well with TensorFlow.

Here is a step-by-step guide to implement and train a CNN for MNIST digit classification using Swift for TensorFlow.
1. Setup Swift for TensorFlow

First, make sure you have Swift for TensorFlow installed. You can find the installation instructions on the Swift for TensorFlow GitHub.
2. Code for MNIST Digit Classification

Below is the Swift code for building and training a CNN model for MNIST digit classification. The MNIST dataset consists of handwritten digits (0-9) and is widely used for benchmarking machine learning models.

import TensorFlow
import PythonKit

// Load MNIST data using PythonKit
let np = Python.import("numpy")
let mnist = Python.import("tensorflow.keras.datasets.mnist")

// Load and preprocess MNIST data
let (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
var xTrainTensor = Tensor<Float>(numpy: np.array(xTrain) / 255.0)!
var xTestTensor = Tensor<Float>(numpy: np.array(xTest) / 255.0)!

// Reshape the data to match the input shape (28x28x1 for CNN)
xTrainTensor = xTrainTensor.reshaped(to: [xTrainTensor.shape[0], 28, 28, 1])
xTestTensor = xTestTensor.reshaped(to: [xTestTensor.shape[0], 28, 28, 1])

// One-hot encode the labels
yTrainTensor = Tensor<Float>(yTrain).oneHot(length: 10)
yTestTensor = Tensor<Float>(yTest).oneHot(length: 10)

// Define the CNN model for MNIST digit classification
struct CNN: Layer {
    var conv1: Conv2D<Float>
    var conv2: Conv2D<Float>
    var conv3: Conv2D<Float>
    var flatten: Flatten<Float>
    var dense1: Dense<Float>
    var dense2: Dense<Float>

    init() {
        conv1 = Conv2D(filterShape: (5, 5, 1, 32), activation: relu)
        conv2 = Conv2D(filterShape: (5, 5, 32, 64), activation: relu)
        conv3 = Conv2D(filterShape: (5, 5, 64, 128), activation: relu)
        flatten = Flatten()
        dense1 = Dense<Float>(inputSize: 128 * 3 * 3, outputSize: 128, activation: relu)
        dense2 = Dense<Float>(inputSize: 128, outputSize: 10, activation: softmax)
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let conv1Out = conv1(input)
        let conv2Out = conv2(conv1Out)
        let conv3Out = conv3(conv2Out)
        let flattened = flatten(conv3Out)
        let dense1Out = dense1(flattened)
        let dense2Out = dense2(dense1Out)
        return dense2Out
    }
}

// Initialize the model
var model = CNN()

// Define optimizer and loss function
let optimizer = Adam(for: model)
let lossFunction = softmaxCrossEntropy(logits:labels:)

// Training loop
for epoch in 1...10 {
    var epochLoss: Float = 0.0

    for i in 0..<xTrainTensor.shape[0] {
        let batchX = xTrainTensor[i...i]
        let batchY = yTrainTensor[i...i]
        
        // Perform forward pass and calculate loss
        let (loss, grads) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let predictions = model(batchX)
            return lossFunction(predictions, batchY)
        }

        // Update the model using the gradients
        optimizer.update(&model, along: grads)
        
        // Accumulate loss for reporting
        epochLoss += loss.scalarized()
    }

    // Print the loss for every epoch
    print("Epoch \(epoch): Loss = \(epochLoss / Float(xTrainTensor.shape[0]))")
}

// Evaluate the model on the test data
let testPredictions = model(xTestTensor)
let testAccuracy = accuracy(predictions: testPredictions, labels: yTestTensor)
print("Test Accuracy: \(testAccuracy)")

3. Code Explanation

    Load MNIST Dataset:
        We use PythonKit to load the MNIST dataset from tensorflow.keras.datasets.mnist.
        The dataset consists of 60,000 training images and 10,000 test images, each 28x28 pixels.

    Preprocessing:
        Normalize the images by dividing by 255.0 to bring the pixel values into the range [0, 1].
        Reshape the data to match the expected shape of the CNN input: [batchSize, 28, 28, 1] (for grayscale images).

    Model Definition (CNN):
        We define a CNN model with three convolutional layers (using Conv2D), each followed by a ReLU activation function.
        After the convolutional layers, the model is flattened using the Flatten layer.
        Two Dense layers are used, the second being the output layer with 10 units (one for each digit) and a softmax activation function to predict probabilities for each class.

    Training Loop:
        We use the Adam optimizer and softmax cross-entropy loss to train the model.
        The model is trained for 10 epochs, with batch processing.

    Evaluation:
        After training, the model is evaluated on the test dataset, and the accuracy is computed.

4. Notes on Swift for TensorFlow

    Swift for TensorFlow allows you to write machine learning code in Swift while leveraging the power of TensorFlow's computational graph.
    The PythonKit library is used to load the MNIST dataset, but you could also implement the data loading and preprocessing in pure Swift if desired.
    This code uses automatic differentiation (@differentiable), which is built into Swift for TensorFlow to compute gradients and update the model parameters.

Conclusion

This code demonstrates how to implement and train a CNN for MNIST digit classification using Swift for TensorFlow. It covers preprocessing, model definition, training, and evaluation of a deep learning model, leveraging Swift's advanced features like automatic differentiation and TensorFlow's deep learning capabilities.

With Swift for TensorFlow, you can easily create efficient deep learning models and run them on macOS or Linux, while also benefiting from Swift's safety, performance, and modern programming features.
