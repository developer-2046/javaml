package javaml.metrics;

public class Precision {

    public static double calculate(int truePositives, int falsePositives) {
        return (double) truePositives / (truePositives + falsePositives);
    }

}
