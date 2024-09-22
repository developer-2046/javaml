package javaml.loss;

import javaml.core.LossFunction;

public class MeanSquaredError implements LossFunction {
    @Override
    public double calculate(double[][] predictions, double[] labels) {
        double totalLoss = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                totalLoss += Math.pow(predictions[i][j] - labels[i], 2);
            }
        }
        return totalLoss / (2 * predictions.length);
    }

    @Override
    public double[][] backward(double[][] predictions, double[] labels) {
        double[][] gradients = new double[predictions.length][predictions[0].length];

        for (int i = 0; i < predictions.length; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                gradients[i][j] = predictions[i][j] - labels[i];
            }
        }
        return gradients;
    }
    }

