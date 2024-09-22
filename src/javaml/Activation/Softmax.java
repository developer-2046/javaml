package javaml.Activation;

import javaml.core.Activation;

public class Softmax implements Activation {

    private double[][] input;


    @Override
    public double[][] forward(double[][] input) {
        this.input = input;
        double[][] output = new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            double sum = 0;
            for (int j = 0; j < input[i].length; j++) {
                output[i][j] = Math.exp(input[i][j]);
                sum += output[i][j];
            }

            for (int j = 0; j < input[i].length; j++) {
                output[i][j] /= sum;
            }
        }
        return output;
    }

    @Override
    public double[][] backward(double[][] gradients) {
       return gradients;
    }
}
