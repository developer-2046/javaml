package javaml.core;

public interface ClassificationModel extends MLModel{

    // Classify outcomes with a specified threshold (e.g., 0.5 for logistic regression)
    int[] classify(double[][] features, double threshold);

    // Evaluate classification javaml.metrics (e.g., accuracy, precision, recall, etc.)
    double evaluateClassification(double[][] features, double[] labels, String metric);

}
