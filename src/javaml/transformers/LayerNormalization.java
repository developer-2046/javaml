package javaml.transformers;

public class LayerNormalization {
    private int dModel;
    private double epsilon = 1e-6;

    public LayerNormalization(int dModel) {
        this.dModel = dModel;
    }

    public double[][] call(double[][] input) {
        double[] mean = new double[input.length];
        double[] variance = new double[input.length];

        // Calculate mean and variance
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                mean[i] += input[i][j];
            }
            mean[i] /= input[0].length;

            for (int j = 0; j < input[0].length; j++) {
                variance[i] += Math.pow(input[i][j] - mean[i], 2);
            }
            variance[i] /= input[0].length;
        }

        // Normalize input
        double[][] normalized = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                normalized[i][j] = (input[i][j] - mean[i]) / Math.sqrt(variance[i] + epsilon);
            }
        }

        return normalized;
    }
}
