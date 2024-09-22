package javaml.algorithms;

public class TreeNode {

    int featureIndex;      // The index of the feature this node splits on
    double threshold;      // The threshold value for splitting
    TreeNode leftChild;    // Left child node (where feature value <= threshold)
    TreeNode rightChild;   // Right child node (where feature value > threshold)
    boolean isLeaf;        // Is this a leaf node?
    double predictedValue; // The predicted value if it's a leaf node

    public TreeNode(int featureIndex, double threshold){
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.isLeaf = false;
    }
    public TreeNode(double predictedValue){
        this.predictedValue = predictedValue;
        this.isLeaf = true;
    }
}
