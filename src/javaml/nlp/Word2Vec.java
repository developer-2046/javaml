package javaml.nlp;

import java.util.*;

public class Word2Vec {
    private int embeddingSize;
    private int windowSize;
    private int minCount;
    private Map<String, Integer> wordToIndex;
    private Map<Integer, String> indexToWord;
    private double[][] W1; // Input to hidden
    private double[][] W2; // Hidden to output

    public Word2Vec(int embeddingSize, int windowSize, int minCount) {
        this.embeddingSize = embeddingSize;
        this.windowSize = windowSize;
        this.minCount = minCount;
        wordToIndex = new HashMap<>();
        indexToWord = new HashMap<>();
    }

    public void fit(List<String> corpus) {
        buildVocabulary(corpus);
        initializeWeights();
        List<TrainingSample> trainingSamples = generateTrainingSamples(corpus);
        train(trainingSamples);
    }

    private void buildVocabulary(List<String> corpus) {
        Map<String, Integer> frequency = new HashMap<>();
        for (String word : corpus) {
            frequency.put(word, frequency.getOrDefault(word, 0) + 1);
        }
        int index = 0;
        for (Map.Entry<String, Integer> entry : frequency.entrySet()) {
            if (entry.getValue() >= minCount) {
                wordToIndex.put(entry.getKey(), index);
                indexToWord.put(index, entry.getKey());
                index++;
            }
        }
    }

    private void initializeWeights() {
        int vocabSize = wordToIndex.size();
        Random rand = new Random();
        W1 = new double[vocabSize][embeddingSize];
        W2 = new double[embeddingSize][vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingSize; j++) {
                W1[i][j] = rand.nextGaussian() * 0.01;
                W2[j][i] = rand.nextGaussian() * 0.01;
            }
        }
    }

    private List<TrainingSample> generateTrainingSamples(List<String> corpus) {
        List<TrainingSample> samples = new ArrayList<>();
        for (int i = 0; i < corpus.size(); i++) {
            String targetWord = corpus.get(i);
            if (!wordToIndex.containsKey(targetWord)) continue;
            int targetIndex = wordToIndex.get(targetWord);
            int start = Math.max(0, i - windowSize);
            int end = Math.min(corpus.size(), i + windowSize + 1);
            for (int j = start; j < end; j++) {
                if (j == i) continue;
                String contextWord = corpus.get(j);
                if (!wordToIndex.containsKey(contextWord)) continue;
                int contextIndex = wordToIndex.get(contextWord);
                samples.add(new TrainingSample(targetIndex, contextIndex));
            }
        }
        return samples;
    }

    private void train(List<TrainingSample> samples) {
        double learningRate = 0.01;
        int epochs = 5;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (TrainingSample sample : samples) {
                // Forward pass
                double[] hiddenLayer = W1[sample.targetIndex];

                double[] outputLayer = new double[W2[0].length];
                for (int i = 0; i < outputLayer.length; i++) {
                    for (int j = 0; j < hiddenLayer.length; j++) {
                        outputLayer[i] += hiddenLayer[j] * W2[j][i];
                    }
                    outputLayer[i] = Math.exp(outputLayer[i]);
                }
                // Softmax normalization
                double sum = 0;
                for (double val : outputLayer) {
                    sum += val;
                }
                for (int i = 0; i < outputLayer.length; i++) {
                    outputLayer[i] /= sum;
                }

                // Calculate error (cross-entropy loss gradient)
                double[] error = new double[outputLayer.length];
                for (int i = 0; i < outputLayer.length; i++) {
                    if (i == sample.contextIndex) {
                        error[i] = outputLayer[i] - 1;
                    } else {
                        error[i] = outputLayer[i];
                    }
                }

                // Backpropagation
                // Update W2
                for (int i = 0; i < W2.length; i++) {
                    for (int j = 0; j < W2[i].length; j++) {
                        W2[i][j] -= learningRate * error[j] * hiddenLayer[i];
                    }
                }

                // Update W1
                for (int i = 0; i < W1[sample.targetIndex].length; i++) {
                    double gradient = 0;
                    for (int j = 0; j < error.length; j++) {
                        gradient += error[j] * W2[i][j];
                    }
                    W1[sample.targetIndex][i] -= learningRate * gradient;
                }
            }
            System.out.println("Epoch " + (epoch + 1) + " completed.");
        }
    }

    public double[] getWordVector(String word) {
        if (wordToIndex.containsKey(word)) {
            return W1[wordToIndex.get(word)];
        } else {
            return null;
        }
    }

    private class TrainingSample {
        public int targetIndex;
        public int contextIndex;

        public TrainingSample(int targetIndex, int contextIndex) {
            this.targetIndex = targetIndex;
            this.contextIndex = contextIndex;
        }
    }
}
