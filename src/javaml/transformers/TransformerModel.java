package javaml.transformers;

public class TransformerModel {
    public TransformerEncoder[] encoders;
    private int numLayers;
    public int dModel;
    private MultiHeadAttention mha;


    public TransformerModel(int numLayers, int dModel, int numHeads, int dff) {
        this.numLayers = numLayers;
        this.dModel = dModel;  // Initialize dModel in the constructor
        this.encoders = new TransformerEncoder[numLayers];
        for (int i = 0; i < numLayers; i++) {
            encoders[i] = new TransformerEncoder(dModel, numHeads, dff);
        }
        this.mha = new MultiHeadAttention(dModel, numHeads);

    }

    public double[][] forward(double[][] input, int seqLen) {
        double[][] output = input;
        for (int i = 0; i < numLayers; i++) {
            output = encoders[i].encode(output, seqLen);
        }
        return output;
    }


    public void updateAttentionWeights(double[][] gradient, double learningRate) {
        mha.updateWq(gradient, learningRate);  // Call MultiHeadAttention's updateWq method
        // Similarly call update methods for Wk, Wv, and Wo
    }
}
