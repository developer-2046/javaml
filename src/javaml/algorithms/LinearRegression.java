package javaml.algorithms;

import javaml.core.MLModel;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;

public class LinearRegression implements MLModel {
    private RealVector theta;
    private double alpha;
    private int iterations;
    private double lambda;
    private boolean useSGD;

    public LinearRegression(double alpha, int iterations, boolean useSGD) {
        this.alpha = alpha;
        this.iterations = iterations;
        this.lambda = 0; // Default no regularization
        this.useSGD = useSGD; // Flag to toggle between SGD and Batch Gradient Descent
    }

    @Override
    public void train(double[][] features, double[] labels) {
        int m = labels.length;
        RealMatrix X = addBiasTerm(features);
        RealVector y = MatrixUtils.createRealVector(labels);
        theta = MatrixUtils.createRealVector(new double[X.getColumnDimension()]);

        // Use SGD or Batch Gradient Descent based on the flag
        if (useSGD) {
            trainWithSGD(X, y, m);
        } else {
            trainWithBatchGradientDescent(X, y, m);
        }
    }

    // Batch Gradient Descent implementation
    private void trainWithBatchGradientDescent(RealMatrix X, RealVector y, int m) {
        for (int iter = 0; iter < iterations; iter++) {
            RealVector prediction = X.operate(theta);
            RealVector errors = prediction.subtract(y);

            RealVector gradients = X.transpose().operate(errors).mapDivide(m);
            RealVector regularization = theta.mapMultiply(lambda / m);

            theta = theta.subtract(gradients.add(regularization).mapMultiply(alpha));
        }
    }

    // Stochastic Gradient Descent (SGD) implementation
    private void trainWithSGD(RealMatrix X, RealVector y, int m) {
        for (int iter = 0; iter < iterations; iter++) {
            for (int i = 0; i < m; i++) {
                RealVector singleFeature = X.getRowVector(i); // Get the i-th sample
                double prediction = singleFeature.dotProduct(theta);
                double error = prediction - y.getEntry(i);

                // Compute gradient and update theta
                RealVector gradient = singleFeature.mapMultiply(error);
                theta = theta.subtract(gradient.mapMultiply(alpha));
            }
        }
    }

    @Override
    public double[] predict(double[][] features) {
        RealMatrix X = addBiasTerm(features);
        return X.operate(theta).toArray();
    }

    @Override
    public double evaluate(double[][] features, double[] labels) {
        double[] predictions = predict(features);
        double sumSquaredErrors = 0.0;

        for (int i = 0; i < labels.length; i++) {
            sumSquaredErrors += Math.pow(predictions[i] - labels[i], 2);
        }
        return sumSquaredErrors / labels.length;
    }

    // Add bias term to the feature matrix
    private RealMatrix addBiasTerm(double[][] features) {
        int m = features.length;
        int n = features[0].length;

        double[][] featuresWithBias = new double[m][n + 1];

        for (int i = 0; i < m; i++) {
            featuresWithBias[i][0] = 1.0;
            System.arraycopy(features[i], 0, featuresWithBias[i], 1, n);
        }
        return MatrixUtils.createRealMatrix(featuresWithBias);
    }

    // K-fold cross-validation method
    public double crossValidate(double[][] features, double[] labels, int k) {
        int foldSize = features.length / k;
        double totalError = 0.0;
        for (int i = 0; i < k; i++) {
            double[][] trainFeatures = extractTrainingFeatures(features, i, foldSize);
            double[] trainLabels = extractTrainingLabels(labels, i, foldSize);

            double[][] testFeatures = extractTestFeatures(features, i, foldSize);
            double[] testLabels = extractTestLabels(labels, i, foldSize);

            this.train(trainFeatures, trainLabels);

            totalError += this.evaluate(testFeatures, testLabels);
        }
        return totalError / k;
    }

    private double[][] extractTrainingFeatures(double[][] features, int foldIndex, int foldSize) {
        int trainSize = features.length - foldSize;
        double[][] trainingFeatures = new double[trainSize][features[0].length];

        int trainIndex = 0;
        for (int i = 0; i < features.length; i++) {
            if (i < foldIndex * foldSize || i > (foldIndex + 1) * foldSize) {
                trainingFeatures[trainIndex++] = features[i];
            }
        }
        return trainingFeatures;
    }

    private double[][] extractTestFeatures(double[][] features, int foldIndex, int foldSize) {
        double[][] testFeatures = new double[foldSize][features[0].length];
        for (int i = 0; i < foldSize; i++) {
            testFeatures[i] = features[foldIndex * foldSize + i];
        }
        return testFeatures;
    }

    private double[] extractTestLabels(double[] labels, int foldIndex, int foldSize) {
        double[] testLabels = new double[foldSize];
        for (int i = 0; i < foldSize; i++) {
            testLabels[i] = labels[foldIndex * foldSize + i];
        }
        return testLabels;
    }

    public double[] extractTrainingLabels(double[] labels, int foldIndex, int foldSize) {
        int trainSize = labels.length - foldSize;
        double[] trainingLabels = new double[trainSize];

        int trainIndex = 0;
        for (int i = 0; i < foldSize; i++) {
            if (i < foldIndex * foldSize || i >= (foldIndex + 1) * foldSize) {
                trainingLabels[trainIndex++] = labels[i];
            }
        }
        return trainingLabels;
    }

    // Save model
    public void saveModel(String filePath) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
        oos.writeObject(theta.toArray());
        oos.close();
    }

    // Load model
    public void loadModel(String filePath) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
        double[] loadedTheta = (double[]) ois.readObject();
        theta = MatrixUtils.createRealVector(loadedTheta);
        ois.close();
    }
}
