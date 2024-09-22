package javaml.Activation;

import javaml.core.Activation;

public class Sigmod implements Activation {

    private double[][] input;

    @Override
    public double[][] forward(double[][] input) {

     this.input = input;
     double[][] outPut = new double[input.length][input[0].length];
      for (int i = 0; i < input.length; i++) {
          for (int j = 0; j < input[i].length; j++) {
              outPut[i][j] = 1 / ( 1 + Math.exp(-input[i][j]));
          }
      }
      return outPut;
    }

    @Override
    public double[][] backward(double[][] gradients) {

        double[][] outputGradients = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                double sigmoidValue = 1 / (1 + Math.exp(-input[i][j]));
                outputGradients[i][j] = gradients[i][j] * sigmoidValue * (1 - sigmoidValue);
            }
        }
        return outputGradients;
    }
}
