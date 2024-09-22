package javaml.algorithms;

public class DecisionTreeUtils {

    // Calculate Gini Impurity for a node


    public static double giniImpurity(double[] labels){
        int n = labels.length;
        int[] classCounts = new int[2]; // assuming binary class count

        for(double label: labels){
            classCounts[(int) label]++;
        }

        double p0 = (double) classCounts[0] / n;
        double p1 = (double) classCounts[1] / n;

        return 1.0 - (p0 * p0 + p1 * p1);
    }

    public static double variance(double[] values){
        double mean = 0.0;
        for(double value: values){
            mean += value;
        }
        mean /= values.length;

        double variance = 0.0;
        for(double value: values){
            variance += Math.pow((value - mean), 2);
        }
        return variance / values.length;
    }
    // Calculate variance reduction after a split (for regression)
    public static double varianceReduction(double[] leftValues, double[] rightValues){

        double totalVariance = variance(concatArrays(leftValues, rightValues));
        double leftVariance = variance(leftValues);
        double rightVariance = variance(rightValues);
        return totalVariance - (leftValues.length * leftVariance + rightValues.length * rightVariance) / (leftValues.length + rightValues.length);
    }
    public static double[] concatArrays(double[] a, double[] b){
        double[] result = new double[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }





}
