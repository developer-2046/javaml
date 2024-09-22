package javaml.preprocessing;

import java.util.Arrays;
import java.util.stream.IntStream;

//principle component analysis
public class PCA {
    private int nComponents;

    public PCA(int nComponents) {
        this.nComponents = nComponents;
    }

    public double[][] fitTransform(double[][] data) {
        // Step 1: Standardize data (mean = 0, variance = 1)
        double[][] standardizedData = standardizeData(data);

        // Step 2: Compute covariance matrix
        double[][] covarianceMatrix = calculateCovarianceMatrix(standardizedData);

        // Step 3: Perform eigen decomposition (Eigenvalues and Eigenvectors)
        double[][] eigenVectors = calculateEigenVectors(covarianceMatrix);

        // Step 4: Select top 'nComponents' eigenvectors and project data
        double[][] transformedData = projectDataOntoEigenVectors(standardizedData, eigenVectors);

        return transformedData;
    }

    private double[][] standardizeData(double[][] data) {
        int nRows = data.length;
        int nCols = data[0].length;

        // Mean and standard deviation for each column
        double[] means = new double[nCols];
        double[] stdDevs = new double[nCols];

        // Calculate column means
        for (int j = 0; j < nCols; j++) {
            double sum = 0;
            for (int i = 0; i < nRows; i++) {
                sum += data[i][j];
            }
            means[j] = sum / nRows;
        }

        // Calculate standard deviations
        for (int j = 0; j < nCols; j++) {
            double sum = 0;
            for (int i = 0; i < nRows; i++) {
                sum += Math.pow(data[i][j] - means[j], 2);
            }
            stdDevs[j] = Math.sqrt(sum / (nRows - 1));
        }

        // Standardize the data
        double[][] standardizedData = new double[nRows][nCols];
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                standardizedData[i][j] = (data[i][j] - means[j]) / stdDevs[j];
            }
        }

        return standardizedData;
    }

    private double[][] calculateCovarianceMatrix(double[][] data) {
        int nRows = data.length;
        int nCols = data[0].length;
        double[][] covarianceMatrix = new double[nCols][nCols];

        // Calculate covariance matrix
        for (int i = 0; i < nCols; i++) {
            for (int j = i; j < nCols; j++) {
                double cov = 0;
                for (int k = 0; k < nRows; k++) {
                    cov += (data[k][i] * data[k][j]);
                }
                cov /= (nRows - 1);
                covarianceMatrix[i][j] = cov;
                covarianceMatrix[j][i] = cov; // Covariance matrix is symmetric
            }
        }
        return covarianceMatrix;
    }

    private double[][] calculateEigenVectors(double[][] covarianceMatrix) {
        // Perform eigen decomposition using a library such as Apache Commons Math or JBLAS.
        // Placeholder code for eigen decomposition

        // Normally you'd use a library here like Apache Commons Math or JAMA for real-world use.
        // You would get eigenvalues and eigenvectors from this.

        // Placeholder for now, returning an identity matrix for simplicity.
        int n = covarianceMatrix.length;
        double[][] eigenVectors = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenVectors[i][j] = i == j ? 1.0 : 0.0;
            }
        }

        return eigenVectors;
    }

    private double[][] projectDataOntoEigenVectors(double[][] data, double[][] eigenVectors) {
        int nRows = data.length;
        int nCols = eigenVectors[0].length;

        // Reduce eigenVectors matrix to top nComponents
        double[][] reducedEigenVectors = new double[nCols][nComponents];
        for (int i = 0; i < nCols; i++) {
            for (int j = 0; j < nComponents; j++) {
                reducedEigenVectors[i][j] = eigenVectors[i][j];
            }
        }

        // Project data onto reduced eigenvectors
        double[][] transformedData = new double[nRows][nComponents];
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nComponents; j++) {
                for (int k = 0; k < nCols; k++) {
                    transformedData[i][j] += data[i][k] * reducedEigenVectors[k][j];
                }
            }
        }

        return transformedData;
    }


}
