package javaml.transformers;

public class PositionalEncoding {

    public double[][] getPositionalEncoding(int seqLen, int dModel) {
        double[][] positionalEncoding = new double[seqLen][dModel];

        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < dModel; i++) {
                if (i % 2 == 0) {
                    positionalEncoding[pos][i] = Math.sin(pos / Math.pow(10000, (2 * i) / (double) dModel));
                } else {
                    positionalEncoding[pos][i] = Math.cos(pos / Math.pow(10000, (2 * i) / (double) dModel));
                }
            }
        }

        return positionalEncoding;
    }
}
