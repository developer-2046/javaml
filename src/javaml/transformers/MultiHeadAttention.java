package javaml.transformers;

import java.util.Arrays;

public class MultiHeadAttention {
    private int dModel;
    private int numHeads;
    private int depth;
    public double[][] Wq;  // Declare Query weights
    public double[][] Wk;  // Declare Key weights
    public double[][] Wv;  // Declare Value weights
    public double[][] Wo;  // Declare Output weights

    public MultiHeadAttention(int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.depth = dModel / numHeads;
        initializeWeights();
    }

    private void initializeWeights() {
        Wq = new double[dModel][depth];  // Initialize the Query weight matrix
        Wk = new double[dModel][depth];  // Initialize the Key weight matrix
        Wv = new double[dModel][depth];  // Initialize the Value weight matrix
        Wo = new double[depth][dModel];  // Initialize the Output weight matrix

        // Random initialization of the weights
        for (double[] row : Wq) Arrays.fill(row, Math.random());
        for (double[] row : Wk) Arrays.fill(row, Math.random());
        for (double[] row : Wv) Arrays.fill(row, Math.random());
        for (double[] row : Wo) Arrays.fill(row, Math.random());
    }

    public double[][] call(double[][] q, double[][] k, double[][] v) {
        // Step 1: Linear projections for Q, K, V
        double[][] query = matMul(q, Wq);
        double[][] key = matMul(k, Wk);
        double[][] value = matMul(v, Wv);

        // Step 2: Scaled dot-product attention
        double[][] attentionScores = matMul(query, transpose(key));
        attentionScores = scaleAndSoftmax(attentionScores);

        // Step 3: Multiply with the value
        double[][] weightedValues = matMul(attentionScores, value);

        // Step 4: Concatenate heads and pass through output weight matrix Wo
        return matMul(weightedValues, Wo);
    }
    public void updateWq(double[][] gradient, double learningRate) {
        for (int i = 0; i < Wq.length; i++) {
            for (int j = 0; j < Wq[0].length; j++) {
                Wq[i][j] -= learningRate * gradient[i][j];  // Update Wq with gradient
            }
        }
    }

    private double[][] scaleAndSoftmax(double[][] matrix) {
        int size = matrix.length;
        double scale = 1 / Math.sqrt(depth);

        // Scale and apply softmax
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] = Math.exp(matrix[i][j] * scale);
                sum += matrix[i][j];
            }
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] /= sum;
            }
        }
        return matrix;
    }

    private double[][] transpose(double[][] matrix) {
        double[][] transposed = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
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
