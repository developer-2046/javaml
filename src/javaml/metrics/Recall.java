package javaml.metrics;

public class Recall {

    public static double calculate(int truePositives, int falseNegatives) {
        return (double) truePositives / (truePositives + falseNegatives);
    }
}
