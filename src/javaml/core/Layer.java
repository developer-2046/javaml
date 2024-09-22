package javaml.core;

public abstract class Layer {

    protected double[][] output;
    //forward pass
    public abstract double[][] forward(double[][] input);
    //backward pass
    public abstract double[][] backward(double[][] gradients);
    //initialize weights, biases, etc.
    public abstract void initialize();

    public double[][] getOutput() {
        return output;
    }

}
