package javaml.algorithms;

public class SplitResult {
    double[][] leftFeatures;
    double[][] rightFeatures;
    double[] leftLabels;
    double[] rightLabels;
    int featureIndex;
    double threshold;

    public SplitResult(double[][] leftFeatures, double[][] rightFeatures, double[] leftLabels, double[] rightLabels, int featureIndex, double threshold) {
        this.leftFeatures = leftFeatures;
        this.rightFeatures = rightFeatures;
        this.leftLabels = leftLabels;
        this.rightLabels = rightLabels;
        this.featureIndex = featureIndex;
        this.threshold = threshold;
    }
}
