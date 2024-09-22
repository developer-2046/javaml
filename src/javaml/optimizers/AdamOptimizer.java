package javaml.optimizers;

import javaml.transformers.TransformerModel;

public class AdamOptimizer {
    private double learningRate;
    private double beta1 = 0.9;
    private double beta2 = 0.999;
    private double epsilon = 1e-8;
    private double[][] m;
    private double[][] v;
    private int t = 0;

    public AdamOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public void initialize(TransformerModel model) {
        // Initialize moment estimates m and v for model parameters
        m = new double[model.dModel][model.dModel];
        v = new double[model.dModel][model.dModel];
    }

    public void update(TransformerModel model, double[][] gradient) {
        t++;

        for (int i = 0; i < gradient.length; i++) {
            for (int j = 0; j < gradient[0].length; j++) {
                // Update first and second moment estimates
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradient[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(gradient[i][j], 2);

                // Bias correction
                double mHat = m[i][j] / (1 - Math.pow(beta1, t));
                double vHat = v[i][j] / (1 - Math.pow(beta2, t));

                // Update weights using Adam's update rule
                double update = learningRate * mHat / (Math.sqrt(vHat) + epsilon);

                // Use TransformerModel to update attention weights (indirect access to Wq)
                model.encoders[0].updateAttentionWeights(gradient, update);
            }
        }
    }
}
