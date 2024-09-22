package javaml.tuning;

import javaml.core.MLModel;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class RandomSearch {
    private Map<String, double[]> paramGrid;
    private int nIterations;
    private int nFolds;

    public RandomSearch(Map<String, double[]> paramGrid, int nIterations, int nFolds) {
        this.paramGrid = paramGrid;
        this.nIterations = nIterations;
        this.nFolds = nFolds;
    }

    public void fit(double[][] data, double[] labels, MLModel model) {
        Random random = new Random();
        // Randomly sample hyperparameter combinations and evaluate model
    }
}
