package javaml.utils;

public class DataPreprocessor {

    public static double[][] standardize (double[][] features) {
        int n = features[0].length;
        double[][] result = new double[features.length][n];
        for (int i = 0; i < n; i++) {
            double mean = 0.0;
            double stdDev = 0.0;
            for (int j = 0; j < features.length; j++) {
                mean += features[j][i];
            }
            mean /= features.length;
            for (int j = 0; j < features.length; j++) {
                stdDev += Math.pow(features[j][i] - mean, 2);
            }
            stdDev = Math.sqrt(stdDev / features.length);
            for (int j = 0; j < features.length; j++) {
                result[j][i] = (features[j][i] - mean ) / stdDev;
            }
        }
        return result;
    }

}
