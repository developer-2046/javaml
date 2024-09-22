package javaml.algorithms;

import javaml.core.ClassificationModel;
import java.util.ArrayList;
import java.util.List;

public class OneVsRestLogisticRegression implements ClassificationModel {

    private List<LogisticRegression> classifiers;
    private int numClasses;

    public OneVsRestLogisticRegression(double alpha, int iterations, double lambda, int numClasses) {
        this.numClasses = numClasses;
        classifiers = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            classifiers.add(new LogisticRegression(alpha, iterations, lambda));
        }
    }

    @Override
    public void train(double[][] features, double[] labels) {
        for (int i = 0; i < numClasses; i++) {
            double[] binaryLabels = new double[labels.length];
            for (int j = 0; j < labels.length; j++) {
                binaryLabels[j] = (labels[j] == i) ? 1 : 0;
            }
            classifiers.get(i).train(features, binaryLabels);
        }
    }

    @Override
    public double[] predict(double[][] features) {
        double[][] probabilities = new double[numClasses][features.length];

        for (int i = 0; i < numClasses; i++) {
            double[] classProbabilities = classifiers.get(i).predict(features);
            for (int j = 0; j < classProbabilities.length; j++) {
                probabilities[i][j] = classProbabilities[j];
            }
        }

        double[] predictions = new double[features.length];
        for (int j = 0; j < features.length; j++) {
            double maxProb = -1;
            int bestClass = -1;
            for (int i = 0; i < numClasses; i++) {
                if (probabilities[i][j] > maxProb) {
                    maxProb = probabilities[i][j];
                    bestClass = i;
                }
            }
            predictions[j] = bestClass;
        }

        return predictions;
    }

    @Override
    public double evaluate(double[][] features, double[] labels) {
        double[] predictions = predict(features);
        int correct = 0;
        for (int i = 0; i < labels.length; i++) {
            if (predictions[i] == labels[i]) {
                correct++;
            }
        }
        return (double) correct / labels.length;
    }

    @Override
    public int[] classify(double[][] features, double threshold) {
        throw new UnsupportedOperationException("Use predict method instead for multiclass classification.");
    }

    @Override
    public double evaluateClassification(double[][] features, double[] labels, String metric) {
        return evaluate(features, labels);  // Extend for other javaml.metrics if needed
    }
}
