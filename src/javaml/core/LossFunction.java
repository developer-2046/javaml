package javaml.core;

public interface LossFunction {
    double calculate(double[][] predictions, double[] labels);
    double[][] backward(double[][] predictions, double[] labels);
}
