package javaml.rl;

import javaml.algorithms.neuralnet.NeuralNetwork;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class DQNAgent {
    private NeuralNetwork qNetwork;
    private NeuralNetwork targetNetwork;
    private double learningRate;
    private double gamma; // Discount factor
    private double epsilon; // Exploration rate
    private double epsilonDecay;
    private double epsilonMin;
    private int stateSize;
    private int actionSize;
    private List<Experience> replayMemory;
    private int batchSize;
    private int targetUpdateFrequency;

    public DQNAgent(int stateSize, int actionSize) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.learningRate = 0.001;
        this.gamma = 0.99;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.995;
        this.epsilonMin = 0.01;
        this.batchSize = 64;
        this.targetUpdateFrequency = 10;
        this.replayMemory = new ArrayList<>();
        initializeNetworks();
    }

    private void initializeNetworks() {
        int[] hiddenLayerSizes = {64, 64};
        qNetwork = new NeuralNetwork(stateSize, hiddenLayerSizes, actionSize, learningRate);
        targetNetwork = new NeuralNetwork(stateSize, hiddenLayerSizes, actionSize, learningRate);
    }

    public int chooseAction(double[] state) {
        Random rand = new Random();
        if (rand.nextDouble() < epsilon) {
            return rand.nextInt(actionSize);
        } else {
            double[] qValues = qNetwork.forward(state);
            return getMaxIndex(qValues);
        }
    }

    private int getMaxIndex(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void storeExperience(Experience exp) {
        replayMemory.add(exp);
        if (replayMemory.size() > 100000) {
            replayMemory.remove(0);
        }
    }

    public void train() {
        if (replayMemory.size() < batchSize) {
            return;
        }
        List<Experience> miniBatch = sampleMiniBatch();
        for (Experience exp : miniBatch) {
            double[] target = qNetwork.forward(exp.state);
            double[] nextQValues = targetNetwork.forward(exp.nextState);
            int bestNextAction = getMaxIndex(nextQValues);
            double targetQValue = exp.reward + gamma * nextQValues[bestNextAction];
            target[exp.action] = targetQValue;
            qNetwork.train(new double[][]{exp.state}, new double[][]{target}, 1);
        }
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }

    private List<Experience> sampleMiniBatch() {
        Random rand = new Random();
        List<Experience> miniBatch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            int index = rand.nextInt(replayMemory.size());
            miniBatch.add(replayMemory.get(index));
        }
        return miniBatch;
    }

    public void updateTargetNetwork() {
        // Copy weights from qNetwork to targetNetwork
        targetNetwork.setWeights(qNetwork.getWeights());
    }

    // Inner class to represent experiences
    private class Experience {
        public double[] state;
        public int action;
        public double reward;
        public double[] nextState;
        public boolean done;

        public Experience(double[] state, int action, double reward, double[] nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }
}
