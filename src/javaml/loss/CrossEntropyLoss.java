package javaml.loss;

import javaml.core.LossFunction;

public class CrossEntropyLoss implements LossFunction {


    @Override
    public double calculate(double[][] predictions, double[] labels) {
        double totalLoss = 0.0;

        for (int i = 0; i < predictions.length; i++) {
            totalLoss -= labels[i] * Math.log(predictions[i][0]);
        }
        return totalLoss / predictions.length;
    }

    @Override
    public double[][] backward(double[][] predictions, double[] labels) {
        double[][] gradients = new double[predictions.length][predictions[0].length];

        for (int i = 0; i < predictions.length; i++) {
            gradients[i][0] = -(labels[i] / predictions[i][0]);
        }
        return gradients;
    }
    public double compute(double[][] predicted, double[] actual) {
        double loss = 0.0;
        for (int i = 0; i < actual.length; i++) {
            loss -= actual[i] * Math.log(predicted[0][i] + 1e-9); // Small epsilon to avoid log(0)
        }
        return loss;
    }
    public double[][] computeGradient(double[][] predicted, double[] actual) {
        double[][] gradient = new double[predicted.length][predicted[0].length];
        for (int i = 0; i < actual.length; i++) {
            gradient[0][i] = predicted[0][i] - actual[i];
        }
        return gradient;
    }
}
