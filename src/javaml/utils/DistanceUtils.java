package javaml.utils;

public class DistanceUtils {

    public static double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            sum += Math.pow(point1[i] - point2[i], 2);
        }
        return Math.sqrt(sum);
    }

    public static boolean isCloseEnough(double[] point1, double[] point2) {
        return euclideanDistance(point1, point2) < tolerance;
    }
}
