package javaml.preprocessing;

public class MinMaxScaler {

    private double min = 0.0;
    private double max = 1.0;
    public MinMaxScaler() {}
    public MinMaxScaler(double min, double max) {
        this.min = min;
        this.max = max;
    }
    public double[][] fitTransform(double[][] data) {
        double[][] scaledData = new double[data.length][data[0].length];
        double[] featureMin = new double[data[0].length];
        double[] featureMax = new double[data.length];

        //finding min max of each feature
        for (int i = 0; i < data[0].length; i++) {
            featureMin[i] = Double.MAX_VALUE;
            featureMax[i] = Double.MIN_VALUE;
            for (int j = 0; j < data.length; j++) {
                if (data[j][i] < featureMin[i]) {
                    featureMin[i] = data[j][i];
                }
                if (data[j][i] > featureMax[i]) {
                    featureMax[i] = data[j][i];
                }
            }

        }
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                scaledData[i][j] = (data[i][j] - featureMin[j]) / (featureMax[j] - featureMin[j]) * (max - min) + min;
            }
        }
        return scaledData;
    }

}
