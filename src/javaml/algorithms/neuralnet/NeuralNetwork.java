package javaml.algorithms.neuralnet;

import java.util.Random;

public class NeuralNetwork {
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int outputSize;

    public double[][][] getWeights() {
        return weights;
    }

    public void setWeights(double[][][] weights) {
        this.weights = weights;
    }

    private double[][][] weights;
    private double learningRate;

    public NeuralNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        weights = new double[hiddenLayerSizes.length + 1][][];

        // Initialize weights between input and first hidden layer
        weights[0] = new double[inputSize][hiddenLayerSizes[0]];
        for (int i = 0; i < weights[0].length; i++) {
            for (int j = 0; j < weights[0][i].length; j++) {
                weights[0][i][j] = rand.nextGaussian();
            }
        }

        // Initialize weights between hidden layers
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            weights[i] = new double[hiddenLayerSizes[i - 1]][hiddenLayerSizes[i]];
            for (int j = 0; j < weights[i].length; j++) {
                for (int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j][k] = rand.nextGaussian();
                }
            }
        }

        // Initialize weights between last hidden layer and output
        weights[hiddenLayerSizes.length] = new double[hiddenLayerSizes[hiddenLayerSizes.length - 1]][outputSize];
        for (int i = 0; i < weights[hiddenLayerSizes.length].length; i++) {
            for (int j = 0; j < weights[hiddenLayerSizes.length][i].length; j++) {
                weights[hiddenLayerSizes.length][i][j] = rand.nextGaussian();
            }
        }
    }

    public double[] forward(double[] input) {
        double[] activations = input;
        for (int i = 0; i < weights.length; i++) {
            double[] nextActivations = new double[weights[i][0].length];
            for (int j = 0; j < weights[i][0].length; j++) {
                double sum = 0;
                for (int k = 0; k < activations.length; k++) {
                    sum += activations[k] * weights[i][k][j];
                }
                nextActivations[j] = ActivationFunction.relu(sum); // Using ReLU as activation
            }
            activations = nextActivations;
        }
        return activations;
    }

    public void train(double[][] inputs, double[][] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Forward pass
                double[] output = forward(inputs[i]);

                // Backward pass
                backpropagate(inputs[i], output, targets[i]);
            }
        }
    }

    private void backpropagate(double[] input, double[] output, double[] target) {
        double[] outputError = new double[output.length];
        double[][][] weightDeltas = new double[weights.length][][];

        // Compute output error
        for (int i = 0; i < output.length; i++) {
            outputError[i] = target[i] - output[i];
        }

        // Initialize weight deltas
        for (int i = 0; i < weights.length; i++) {
            weightDeltas[i] = new double[weights[i].length][weights[i][0].length];
        }

        // Backpropagation through layers
        double[] layerError = outputError;
        for (int layer = weights.length - 1; layer >= 0; layer--) {
            double[] previousLayerActivations = layer == 0 ? input : forward(input); // Use input for first layer

            for (int i = 0; i < weights[layer][0].length; i++) {
                for (int j = 0; j < previousLayerActivations.length; j++) {
                    weightDeltas[layer][j][i] = learningRate * layerError[i] * ActivationFunction.reluDerivative(previousLayerActivations[j]);
                }
            }

            // Update weights
            for (int i = 0; i < weights[layer].length; i++) {
                for (int j = 0; j < weights[layer][i].length; j++) {
                    weights[layer][i][j] += weightDeltas[layer][i][j];
                }
            }
        }
    }
}
