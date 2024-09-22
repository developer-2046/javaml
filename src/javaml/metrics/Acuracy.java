package javaml.metrics;

public class Acuracy {

    public double calculate(double[][] predictions, double[] labels){
        int correct = 0;

        for(int i = 0; i<predictions.length; i++){

            int predictedClass = argMax(predictions[i]);
            if(predictedClass == (int) labels[i]){
                correct++;
            }

        }
        return (double) correct / (double) predictions.length;

    }
    private int argMax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}
