package javaml.transformers;

public class FeedForwardNetwork {
    private int dModel;
    private int dff;  // Number of hidden units in the feed-forward network
    private double[][] W1;
    private double[][] W2;

    public FeedForwardNetwork(int dModel, int dff) {
        this.dModel = dModel;
        this.dff = dff;
        initializeWeights();
    }

    private void initializeWeights() {
        W1 = new double[dModel][dff];
        W2 = new double[dff][dModel];

        // Random initialization (for illustration)
        for (double[] row : W1) {
            for (int i = 0; i < dff; i++) row[i] = Math.random();
        }
        for (double[] row : W2) {
            for (int i = 0; i < dModel; i++) row[i] = Math.random();
        }
    }

    public double[][] call(double[][] input) {
        double[][] hiddenLayer = relu(matMul(input, W1));
        return matMul(hiddenLayer, W2);
    }

    private double[][] relu(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[i][j] = Math.max(0, matrix[i][j]);
            }
        }
        return result;
    }

    private double[][] matMul(double[][] A, double[][] B) {
        double[][] result = new double[A.length][B[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                for (int k = 0; k < B.length; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
}
