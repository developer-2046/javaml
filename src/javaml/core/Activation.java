package javaml.core;

public interface Activation {

    double[][] forward(double[][] input);
    double[][] backward(double[][] gradients);
}
