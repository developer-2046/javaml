package javaml.algorithms;

import javaml.core.ClassificationModel;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import java.io.*;
import java.util.Random;

public class LogisticRegression implements ClassificationModel {

    private RealVector theta;
    private double alpha; // learning rate
    private int iterations; // number of iterations
    private double lambda; // regularization parameter

    public LogisticRegression(double alpha, int iterations, double lambda) {
        this.alpha = alpha;
        this.iterations = iterations;
        this.lambda = lambda;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public void train(double[][] features, double[] labels) {
        int m = labels.length;
        RealMatrix X = MatrixUtils.createRealMatrix(addBiasTerm(features));
        RealVector y = MatrixUtils.createRealVector(labels);
        theta = MatrixUtils.createRealVector(new double[X.getColumnDimension()]);
        Random random = new Random();

        // SGD: Loop for a number of iterations
        for (int iter = 0; iter < iterations; iter++) {
            // Shuffle the data at each iteration
            for (int i = 0; i < m; i++) {
                // Randomly pick an index
                int randIndex = random.nextInt(m);
                double[] xi = X.getRow(randIndex);
                RealVector xiVec = MatrixUtils.createRealVector(xi);
                double yi = y.getEntry(randIndex);

                // Calculate the prediction for the single sample
                double z = theta.dotProduct(xiVec);
                double prediction = sigmoid(z);

                // Calculate the error
                double error = prediction - yi;

                // Update the gradient and theta
                RealVector gradient = xiVec.mapMultiply(error);
                RealVector regularization = theta.mapMultiply(lambda / m);
                regularization.setEntry(0, 0); // No regularization for the bias term

                // Update theta (SGD step)
                theta = theta.subtract(gradient.add(regularization).mapMultiply(alpha));
            }
        }
    }

    @Override
    public double[] predict(double[][] features) {
        RealMatrix X = MatrixUtils.createRealMatrix(addBiasTerm(features));
        RealVector z = X.operate(theta);
        return z.map(this::sigmoid).toArray();
    }

    @Override
    public double evaluate(double[][] features, double[] labels) {
        return evaluateClassification(features, labels, "accuracy");
    }

    @Override
    public int[] classify(double[][] features, double threshold) {
        double[] probabilities = predict(features);
        int[] classifications = new int[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            classifications[i] = probabilities[i] >= threshold ? 1 : 0;
        }
        return classifications;
    }

    private double[][] addBiasTerm(double[][] features) {
        double[][] X_new = new double[features.length][features[0].length + 1];
        for (int i = 0; i < features.length; i++) {
            X_new[i][0] = 1.0;
            System.arraycopy(features[i], 0, X_new[i], 1, features[i].length);
        }
        return X_new;
    }

    @Override
    public double evaluateClassification(double[][] features, double[] labels, String metric) {
        int[] predictions = classify(features, 0.5);
        int correct = 0;
        for (int i = 0; i < labels.length; i++) {
            if (predictions[i] == labels[i]) correct++;
        }
        if (metric.equals("accuracy")) return (double) correct / labels.length;
        throw new UnsupportedOperationException("Metric not implemented: " + metric);
    }

    public void saveModel(String filePath) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
        oos.writeObject(theta.toArray());
        oos.close();
    }

    public void loadModel(String filePath) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
        double[] loadedTheta = (double[]) ois.readObject();
        theta = MatrixUtils.createRealVector(loadedTheta);
        ois.close();
    }
}

