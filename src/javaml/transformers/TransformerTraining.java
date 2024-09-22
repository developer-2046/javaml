package javaml.transformers;

import javaml.loss.CrossEntropyLoss;
import javaml.nlp.DataPreparation.TrainingSample;
import javaml.optimizers.AdamOptimizer;


import java.util.List;

public class TransformerTraining {
    private TransformerModel transformer;
    private AdamOptimizer optimizer;
    private CrossEntropyLoss lossFunction;

    public TransformerTraining(TransformerModel transformer, AdamOptimizer optimizer, CrossEntropyLoss lossFunction) {
        this.transformer = transformer;
        this.optimizer = optimizer;
        this.lossFunction = lossFunction;
        this.optimizer.initialize(transformer);  // Initialize the optimizer with model parameters
    }

    public void train(List<TrainingSample> samples, int epochs, int seqLen) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;

            for (TrainingSample sample : samples) {
                double[][] inputVectors = convertToVectors(sample.inputSeq);
                double[] targetVector = convertToVector(sample.target);

                double[][] output = transformer.forward(inputVectors, seqLen);
                double loss = lossFunction.compute(output, targetVector);
                epochLoss += loss;

                double[][] gradients = lossFunction.computeGradient(output, targetVector);
                optimizer.update(transformer, gradients);  // Updated to pass gradients
            }

            System.out.println("Epoch " + (epoch + 1) + " completed with loss: " + (epochLoss / samples.size()));
        }
    }


    private double[][] convertToVectors(List<String> sequence) {
        // Placeholder for converting tokens to embeddings
        return new double[sequence.size()][transformer.dModel];
    }

    private double[] convertToVector(String target) {
        // Placeholder for converting target word to a vector
        return new double[transformer.dModel];
    }
}
