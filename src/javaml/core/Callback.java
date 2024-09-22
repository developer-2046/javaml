package javaml.core;

public abstract class Callback {
    public void onEpochBegin(int epoch) {}
    public void onEpochEnd(int epoch, double loss) {}
    public void onBatchBegin(int batch) {}
    public void onBatchEnd(int batch, double batchLoss) {}
}
