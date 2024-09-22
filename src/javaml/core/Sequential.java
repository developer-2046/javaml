package javaml.core;

import java.util.ArrayList;
import java.util.List;

;

 public class Sequential {

   private List<Layer> layers;
   private Optimizer optimizer;
   private LossFunction lossFunction;

   public Sequential(){
    this.layers = new ArrayList<>();
    }

    public void add(Layer layer){
       this.layers.add(layer);
    }

    public void compile(Optimizer optimizer, LossFunction lossFunction){
       this.optimizer = optimizer;
       this.lossFunction = lossFunction;

       //layer inititilization
        for (Layer layer : layers) {
            layer.initialize();
        }
    }
    public void fit(double[][] features, double[] labels, int epochs, int batchSize){
       for (int epoch =0; epoch < epochs; epoch++) {

           //mini-bacth gradient descent
           for (int i =0; i < features.length; i += batchSize){
               int end = Math.min(i+batchSize, features.length);
               double[][] batchFeatures = getBatch(features, i, end);
               double[] batchLabels = getBatch(labels, i, end);

               double[][] predictions = forward(batchFeatures);

               double loss = lossFunction.calculate(predictions, batchLabels);

               backpropagate(batchLabels);

               optimizer.update(layers);

           }
       }
    }

    private double[][] forward(double[][] features){
       double[][] output = features;
       for (Layer layer : layers) {
          output =  layer.forward(output);
       }
       return output;
    }

    private void backpropagate(double[] labels){
       double[][] gradients = lossFunction.backward(layers.get(layers.size() -1).getOutput(), labels);
       for (int i = layers.size() - 1; i >= 0; i--) {
           gradients = layers.get(i).backward(gradients);
       }
    }
     private double[][] getBatch(double[][] data, int start, int end) {
         double[][] batch = new double[end - start][data[0].length];
         System.arraycopy(data, start, batch, 0, end - start);
         return batch;
     }

     private double[] getBatch(double[] data, int start, int end) {
         double[] batch = new double[end - start];
         System.arraycopy(data, start, batch, 0, end - start);
         return batch;
     }


 }
