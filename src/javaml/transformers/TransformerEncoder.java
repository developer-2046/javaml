package javaml.transformers;

public class TransformerEncoder {
    private int dModel;
    private int numHeads;
    private int dff;  // Hidden size for the Feed-Forward Network
    private MultiHeadAttention mha;
    private FeedForwardNetwork ffn;
    private LayerNormalization layerNorm1;
    private LayerNormalization layerNorm2;
    private PositionalEncoding posEnc;

    public TransformerEncoder(int dModel, int numHeads, int dff) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dff = dff;
        this.mha = new MultiHeadAttention(dModel, numHeads);
        this.ffn = new FeedForwardNetwork(dModel, dff);
        this.layerNorm1 = new LayerNormalization(dModel);
        this.layerNorm2 = new LayerNormalization(dModel);
        this.posEnc = new PositionalEncoding();
    }

    public double[][] encode(double[][] input, int seqLen) {
        // Step 1: Add positional encoding to input
        double[][] posEncoding = posEnc.getPositionalEncoding(seqLen, dModel);
        input = add(input, posEncoding);

        // Step 2: Multi-Head Attention
        double[][] attnOutput = mha.call(input, input, input);  // No direct interaction with Wq, Wk, etc.
        double[][] normAttnOutput = layerNorm1.call(add(input, attnOutput));

        // Step 3: Feed-Forward Network
        double[][] ffnOutput = ffn.call(normAttnOutput);
        return layerNorm2.call(add(normAttnOutput, ffnOutput));
    }
    public void updateAttentionWeights(double[][] gradient, double learningRate) {
        mha.updateWq(gradient, learningRate);  // Call MultiHeadAttention's updateWq method
        // Similarly call update methods for Wk, Wv, and Wo
    }

    // Helper method for element-wise addition
    private double[][] add(double[][] a, double[][] b) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }
}

