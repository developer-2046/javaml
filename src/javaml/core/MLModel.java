package javaml.core;

public interface MLModel {

    void train(double[][] features, double[] labels);
    double[] predict(double[][] features);
    double evaluate(double[][] feature, double[] labels);
}
