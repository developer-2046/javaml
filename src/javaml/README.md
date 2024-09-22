
# JavaML - Machine Learning Library in Java

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**JavaML** is a comprehensive, flexible, and lightweight machine learning library written in Java. The library aims to provide a wide range of machine learning algorithms for both beginners and advanced developers. JavaML is inspired by libraries such as Keras and Scikit-learn and is designed to make building and training machine learning models as intuitive as possible.

## Features

- **Regression Models**:
  - Linear Regression
  - Logistic Regression
  - Regularization Support

- **Classification Models**:
  - Decision Trees
  - One-vs-Rest Logistic Regression

- **Unsupervised Learning**:
  - K-Means Clustering

- **Neural Networks**:
  - Feedforward Neural Networks
  - Multi-Layer Perceptrons (MLPs)
  - Dropout, Dense, Batch Normalization Layers
  - Transformers (Multi-Head Attention, Feedforward Layers, Positional Encoding)

- **Reinforcement Learning**:
  - Q-Learning
  - Deep Q-Network (DQN)

- **Natural Language Processing**:
  - Tokenizer
  - Word2Vec

- **Model Evaluation**:
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - Accuracy

- **Optimizers**:
  - Stochastic Gradient Descent (SGD)
  - Adam Optimizer

- **Hyperparameter Tuning**:
  - Grid Search
  - Random Search

## Project Structure

```
|-- src/
|   |-- algorithms/
|       |-- LinearRegression.java
|       |-- LogisticRegression.java
|       |-- DecisionTree.java
|       |-- KMeans.java
|   |-- transformers/
|       |-- TransformerEncoder.java
|       |-- MultiHeadAttention.java
|       |-- FeedForwardNetwork.java
|   |-- nlp/
|       |-- Tokenizer.java
|       |-- Word2Vec.java
|   |-- rl/
|       |-- QLearning.java
|       |-- DQNAgent.java
|   |-- metrics/
|       |-- Precision.java
|       |-- Recall.java
|       |-- F1Score.java
|       |-- Accuracy.java
|   |-- utils/
|       |-- CrossValidation.java
|   |-- optimizers/
|       |-- SGD.java
|       |-- AdamOptimizer.java
|-- README.md
|-- LICENSE
```

## Installation

To use **JavaML**, clone the repository and include the source files in your Java project.

```bash
git clone https://github.com/developer-2046/javaml.git
```

### Prerequisites

Ensure you have the following setup:

- Java 8 or above
- Maven (for dependency management)

## Usage

### 1. **Linear Regression**

```java
import javaml.algorithms.LinearRegression;

public class Main {
    public static void main(String[] args) {
        double[][] X = { {1, 1}, {2, 2}, {3, 3}, {4, 4} };
        double[] y = { 2, 4, 6, 8 };

        LinearRegression lr = new LinearRegression();
        lr.fit(X, y);
        
        double[] predictions = lr.predict(X);
        for (double pred : predictions) {
            System.out.println(pred);
        }
    }
}
```

### 2. **K-Means Clustering**

```java
import javaml.algorithms.KMeans;

public class Main {
    public static void main(String[] args) {
        double[][] data = { {1, 1}, {2, 1}, {4, 3}, {5, 4} };
        int K = 2;
        
        KMeans kmeans = new KMeans(K, 100);
        kmeans.fit(data);
        
        int[] clusters = kmeans.predict(data);
        for (int cluster : clusters) {
            System.out.println(cluster);
        }
    }
}
```

### 3. **Transformer Model**

```java
import javaml.transformers.TransformerModel;

public class Main {
    public static void main(String[] args) {
        TransformerModel transformer = new TransformerModel(6, 512, 8, 2048);
        double[][] input = { /* input data */ };
        double[][] output = transformer.forward(input, input.length);

        // Use output for further tasks
    }
}
```

### 4. **Q-Learning (Reinforcement Learning)**

```java
import javaml.rl.QLearning;

public class Main {
    public static void main(String[] args) {
        QLearning agent = new QLearning(10, 2, 0.1, 0.99, 0.9);
        
        // Define your states, actions, and rewards
        // Call agent.chooseAction(state) to get an action
        // Call agent.updateQTable(state, action, reward, nextState) after each step
    }
}
```

## Documentation

For detailed usage, you can check the individual classes and their respective methods in the `/src` directory. Each class is documented with the appropriate comments to guide you through the process.

## Contributing

Contributions are welcome! Feel free to submit a pull request, open an issue, or discuss a feature request. Please ensure that your contributions are well-tested and documented.

1. Fork the project
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Keras**: For inspiring the modular design of the library
- **Scikit-Learn**: For the structured approach to model building
- **TensorFlow**: For advanced neural network and deep learning capabilities
