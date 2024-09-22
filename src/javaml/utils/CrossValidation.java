package javaml.utils;

import javaml.core.MLModel;
import java.util.ArrayList;
import java.util.List;

public class CrossValidation {

    public static double crossValidate(MLModel model, double[][] features, double[] labels, int kFolds) {
        int foldSize = features.length / kFolds;
        List<Double> scores = new ArrayList<>();

        for (int i = 0; i < kFolds; i++) {
            // Split into training and validation sets
            double[][] trainFeatures = new double[features.length - foldSize][];
            double[] trainLabels = new double[labels.length - foldSize];
            double[][] valFeatures = new double[foldSize][];
            double[] valLabels = new double[foldSize];

            // Fill train and validation sets
            int trainIndex = 0, valIndex = 0;
            for (int j = 0; j < features.length; j++) {
                if (j >= i * foldSize && j < (i + 1) * foldSize) {
                    valFeatures[valIndex] = features[j];
                    valLabels[valIndex++] = labels[j];
                } else {
                    trainFeatures[trainIndex] = features[j];
                    trainLabels[trainIndex++] = labels[j];
                }
            }

            // Train the model
            model.train(trainFeatures, trainLabels);

            // Evaluate on validation set
            double score = model.evaluate(valFeatures, valLabels);
            scores.add(score);
        }

        // Return average score
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
}
