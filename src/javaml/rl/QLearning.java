package javaml.rl;

import java.util.Random;

public class QLearning {
    private double[][] qTable;
    private double learningRate;
    private double discountFactor;
    private double epsilon;  // For exploration vs. exploitation
    private int nStates;
    private int nActions;

    public QLearning(int nStates, int nActions, double learningRate, double discountFactor, double epsilon) {
        this.nStates = nStates;
        this.nActions = nActions;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.epsilon = epsilon;
        this.qTable = new double[nStates][nActions];
        initializeQTable();
    }

    private void initializeQTable() {
        Random rand = new Random();
        for (int i = 0; i < nStates; i++) {
            for (int j = 0; j < nActions; j++) {
                qTable[i][j] = rand.nextDouble();  // Random initialization
            }
        }
    }

    public int chooseAction(int state) {
        Random rand = new Random();
        if (rand.nextDouble() < epsilon) {
            // Explore: choose a random action
            return rand.nextInt(nActions);
        } else {
            // Exploit: choose the best known action
            return getMaxQAction(state);
        }
    }

    private int getMaxQAction(int state) {
        int bestAction = 0;
        double maxQValue = qTable[state][0];
        for (int i = 1; i < qTable[state].length; i++) {
            if (qTable[state][i] > maxQValue) {
                maxQValue = qTable[state][i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    public void updateQTable(int state, int action, int nextState, double reward) {
        double bestFutureQValue = qTable[nextState][getMaxQAction(nextState)];
        qTable[state][action] = qTable[state][action] + learningRate * (reward + discountFactor * bestFutureQValue - qTable[state][action]);
    }

    public double[][] getQTable() {
        return qTable;
    }
}
