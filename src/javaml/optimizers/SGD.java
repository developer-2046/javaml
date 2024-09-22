package javaml.optimizers;

import javaml.core.Layer;
import javaml.core.Optimizer;
import javaml.layers.Dense;

import java.util.List;

public class SGD extends Optimizer {
    private double learningRate;
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(List<Layer> layers) {
        for (Layer layer0 : layers) {
            for (Layer layer : layers) {
                if (layer instanceof Dense) {
                    Dense denseLayer = (Dense) layer;

                    double[][] weightGradients = denseLayer.getWeightGradients();
                    double[] biasGradients = denseLayer.getBiasGradients();

                    // Update weights
                    double[][] weights = denseLayer.getWeights();
                    for (int i = 0; i < weights.length; i++) {
                        for (int j = 0; j < weights[i].length; j++) {
                            weights[i][j] -= learningRate * weightGradients[i][j];
                        }
                    }
                    denseLayer.setWeights(weights);

                    // Update biases
                    double[] biases = denseLayer.getBiases();
                    for (int i = 0; i < biases.length; i++) {
                        biases[i] -= learningRate * biasGradients[i];
                    }
                    denseLayer.setBiases(biases);
                }
            }
        }

    }
}
