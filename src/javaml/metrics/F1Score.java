package javaml.metrics;

public class F1Score {

    public static double calculate(double precision, double recall){
        return 2*((precision * recall) / (precision + recall));

    }

}
