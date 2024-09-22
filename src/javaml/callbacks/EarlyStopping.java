package javaml.callbacks;

import javaml.core.Callback;

public class EarlyStopping extends Callback {

    private int patience;
    private int wait = 0;
    private double bestLoss = Double.MAX_VALUE;
    private boolean stopped = false;

    public EarlyStopping(int patience) {
        this.patience = patience;
    }
    @Override
    public void onEpochEnd(int epoch, double loss){
        if(loss<bestLoss){
            bestLoss = loss;
            wait = 0;
        }
        else {
            wait++;
            if(wait>patience){
                stopped = true;
            }
        }
    }

    public boolean shouldStop(){
        return stopped;
    }

}
