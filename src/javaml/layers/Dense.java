package javaml.layers;

import javaml.core.Activation;
import javaml.core.Layer;

import java.util.Random;

public class Dense extends Layer {

    private double[][] weights;
    private double[] biases;
    private double[][] input;
    private int inputSize;
    private int outputSize;
    private double[][] weightGradients;
    private double[] biasGradients;
    private Activation activation;

    public Dense(int inputSize, int outputSize, Activation activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
    }


    @Override
    public double[][] forward(double[][] input) {

        this.input = input;
        double[][] output = new double[input.length][outputSize];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < outputSize; j++) {
                output[i][j] = 0.0;
                for (int k = 0; k < inputSize; k++) {
                    output[i][j] += input[i][k] * input[k][j];
                }
                output[i][j] += biases[i];
            }
        }
        if (activation != null) {
            output = activation.forward(output);
        }

        this.output = output;
        return output;

    }

    @Override
    public double[][] backward(double[][] gradients) {
        if (activation != null) {
            gradients = activation.backward(gradients);
        }
         weightGradients = new double[inputSize][outputSize];
         biasGradients = new double[outputSize];
        double[][] inputGradients = new double[input.length][inputSize];

        //compute gradients for weights, bias and inputs
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < outputSize; j++) {
                biasGradients[j] += gradients[i][j];
                for (int k = 0; k < inputSize; k++) {
                    weightGradients[j][i] += gradients[i][k] * input[i][j];
                    inputGradients[i][k] += gradients[k][j] * input[i][j];
                }
            }
        }

        return inputGradients;
    }
    public double[][] getWeightGradients() {
        return weightGradients;
    }
    public double[] getBiasGradients() {
        return biasGradients;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public void setWeights(double[][] newWeights) {
        this.weights = newWeights;
    }

    public void setBiases(double[] newBiases) {
        this.biases = newBiases;
    }
    @Override
    public void initialize() {

        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];
        Random r = new Random();
        // random weight initialize
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = r.nextGaussian() * 0.01;
            }
        }

        //initialize biases to zero
        for (int i = 0; i < outputSize; i++) {
            biases[i] = 0.0;
        }

    }
}
