package javaml.algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DecisionTree {

    private int maxDepth;
    private int minSamplesSplit;
    private boolean isRegression;
    private TreeNode rootNode;

    public DecisionTree(int maxDepth, int minSamplesSplit, boolean isRegression) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.isRegression = isRegression;

    }
    public void train(double[][] features, double[] labels) {
        rootNode = buildTree(features, labels, 0);
    }
    public TreeNode buildTree(double[][] features, double[] labels, int depth) {
        if (depth >= maxDepth || labels.length <= minSamplesSplit) {
            return new TreeNode(calculateLeafValue(labels));
        }

        // Find the best split
        SplitResult bestSplit = findBestSplit(features, labels);
        if (bestSplit == null) {
            return new TreeNode(calculateLeafValue(labels)); // Leaf node
        }

        // Split data into left and right branches
        double[][] leftFeatures = bestSplit.leftFeatures;
        double[][] rightFeatures = bestSplit.rightFeatures;
        double[] leftLabels = bestSplit.leftLabels;
        double[] rightLabels = bestSplit.rightLabels;

        // Create child nodes
        TreeNode leftChild = buildTree(leftFeatures, leftLabels, depth + 1);
        TreeNode rightChild = buildTree(rightFeatures, rightLabels, depth + 1);

        // Create internal node
        TreeNode node = new TreeNode(bestSplit.featureIndex, bestSplit.threshold);
        node.leftChild = leftChild;
        node.rightChild = rightChild;
        return node;
    }

    private double calculateLeafValue(double[] labels){
        if(isRegression){
            return Arrays.stream(labels).average().orElse(0.0);
        }
        else {
            int[] counts = new int[2]; // binary classification
            for(double label : labels){
                counts[(int)label]++;
            }
            return (counts[0] > counts[1]) ? 0 : 1;
        }
    }
    private SplitResult findBestSplit(double[][] features, double[] labels) {
        int nFeatures = features[0].length;
        double bestGain = -Double.MAX_VALUE;
        SplitResult bestSplit = null;

        for (int feature = 0; feature < nFeatures; feature++) {
            for (double threshold : getUniqueValues(features, feature)) {
                SplitResult split = splitData(features, labels, feature, threshold);
                double gain = calculateGain(split, labels);

                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = split;
                }
            }
        }
        return bestSplit;
    }
    // Get unique values of a feature (used to test thresholds)
    private double[] getUniqueValues(double[][] features, int featureIndex) {
        return Arrays.stream(features).mapToDouble(f -> f[featureIndex]).distinct().toArray();
    }
    // Split data by a given feature and threshold
    private SplitResult splitData(double[][] features, double[] labels, int featureIndex, double threshold) {
        // Left and right lists for features and labels
        List<double[]> leftFeaturesList = new ArrayList<>();
        List<double[]> rightFeaturesList = new ArrayList<>();
        List<Double> leftLabelsList = new ArrayList<>();
        List<Double> rightLabelsList = new ArrayList<>();

        // Iterate through all samples and split based on the threshold
        for (int i = 0; i < features.length; i++) {
            if (features[i][featureIndex] <= threshold) {
                leftFeaturesList.add(features[i]);
                leftLabelsList.add(labels[i]);
            } else {
                rightFeaturesList.add(features[i]);
                rightLabelsList.add(labels[i]);
            }
        }

        // Convert the lists back to arrays
        double[][] leftFeatures = leftFeaturesList.toArray(new double[0][0]);
        double[][] rightFeatures = rightFeaturesList.toArray(new double[0][0]);
        double[] leftLabels = leftLabelsList.stream().mapToDouble(Double::doubleValue).toArray();
        double[] rightLabels = rightLabelsList.stream().mapToDouble(Double::doubleValue).toArray();

        // Return the SplitResult object
        return new SplitResult(leftFeatures, rightFeatures, leftLabels, rightLabels, featureIndex, threshold);
    }
    private double calculateGain(SplitResult split, double[] labels){
        double totalVariance = DecisionTreeUtils.variance(labels);
        double leftVariance = DecisionTreeUtils.variance(split.leftLabels);
        double rightVariance = DecisionTreeUtils.variance(split.rightLabels);

        double weightedAverageVariance = (split.leftLabels.length * leftVariance * split.rightLabels.length * rightVariance ) / labels.length;

        return totalVariance - weightedAverageVariance;
    }

    //prediction for a single sample
    public double predictSample(TreeNode node, double[] features){
        if (node.isLeaf) return node.predictedValue;
        if (features[node.featureIndex] <= node.threshold) {
            return predictSample(node.leftChild, features);
        } else {
            return predictSample(node.rightChild, features);
        }
    }
    //prediction for a set of samples
    public double[] predict(double[][] features){
        double[] predictions = new double[features.length];

        for (int i = 0; i < features.length; i++) {
            predictions[i] = predictSample(rootNode, features[i]);
        }
        return predictions;
    }


}
