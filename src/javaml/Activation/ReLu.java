package javaml.Activation;

import javaml.core.Activation;

// Rectified Linear Unit
public class ReLu implements Activation {

    private double[][] input;

    @Override
    public double[][] forward(double[][] input) {

        this.input = input;
        double[][] output = new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                output[i][j] = Math.max(0, input[i][j]);
            }
        }
        return output;
    }

    @Override
    public double[][] backward(double[][] gradients) {

        double[][] outputGradients = new double[gradients.length][gradients[0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                outputGradients[i][j] = input[i][j] > 0 ? gradients[i][j] : 0;
            }
        }
          return outputGradients;
    }
}
