package javaml.algorithms.unsupervised;

import javaml.utils.DistanceUtils;

import java.util.ArrayList;

import java.util.List;
import java.util.Random;

public class KMeans {

    private int K;
    private int maxIterations;
    private List<double[]> centroids;

    public KMeans(int K, int maxIterations) {
        this.K = K;
        this.maxIterations = maxIterations;
        this.centroids = new ArrayList<>();
    }
    public void fit(double[][] data){
     initializeCentroids(data);
     for(int i = 0; i < maxIterations; i++){
         int[] clusterAssignments = assignCulture(data);
         List<double[]> newCentroids = updateCentroids(data, clusterAssignments);
         if (hasConverged(centroids, newCentroids)) {
             break;
         }
         centroids = newCentroids;
     }
    }
    public void initializeCentroids(double[][] data){
        Random rand = new Random();
        centroids.clear();
        for (int i = 0; i < K; i++) {
            centroids.add(data[rand.nextInt(data.length)]);
        }
    }
    private int[] assignCulture(double[][] data){
        int[] assignments = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            assignments[i] = findClosestCentroid(data[i]);
        }
        return assignments;
    }
    private int findClosestCentroid(double[] dataPoint){
        double minDistance = Double.MAX_VALUE;
        int closestIndex = 1;
        for (int i = 0; i < centroids.size(); i++) {
            double distance = DistanceUtils.euclideanDistance(dataPoint, centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                closestIndex = i;
            }
        }
        return closestIndex;
    }
    private List<double[]> updateCentroids(double[][] data, int[] clusterAssignments){
        List<double[]> newCentroids = new ArrayList<>();
        for (int i = 0; i < K; i++) {
            List<double[]> pointsInCluster = new ArrayList<>();
            for (int j = 0; j < data.length; j++) {
                if (clusterAssignments[j] == i) {pointsInCluster.add(data[j]); }
            }
             newCentroids.add(meanOfPoints(pointsInCluster));
        }
       return newCentroids;
    }
    private double[] meanOfPoints (List<double[]> points){
        double[] mean = new double[points.get(0).length];
        for(double[] point : points){
            for (int i = 0; i < point.length; i++) {
                mean[i] += point[i];
            }        }
        for(int i = 0; i < mean.length; i++){
            mean[i] /= points.size();
        }
        return mean;
    }
    private boolean hasConverged(List<double[]> oldCentroids, List<double[]> newCentroids) {
        for (int i = 0; i < oldCentroids.size(); i++) {
            if (!DistanceUtils.isCloseEnough(oldCentroids.get(i), newCentroids.get(i))) {
                return false;
            }
        }
        return true;
    }
}
