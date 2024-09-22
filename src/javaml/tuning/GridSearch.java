package javaml.tuning;

import javaml.core.MLModel;

import java.util.HashMap;
import java.util.Map;

public class GridSearch {
    private Map<String, double[]> paramGrid;
    private int nFolds;

    public GridSearch(Map<String, double[]> paramGrid, int nFolds) {
        this.paramGrid = paramGrid;
        this.nFolds = nFolds;
    }

    public void fit(double[][] data, double[] labels, MLModel model) {
        // Implement cross-validation based grid search
        // Iterate over all parameter combinations
        // Evaluate the model performance for each combination
    }
}
