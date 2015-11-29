
import ann.ANNOptions;
import ann.MultiLayerPerceptron;
import ann.WekaUtil;
import javax.rmi.CORBA.Util;
import weka.classifiers.Classifier;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Fahmi
 */
public class Main {
    public static void main(String[] args) throws Exception{
        //set Topology
        ANNOptions.bias = 1;
        ANNOptions.biasWeight = 0.1;
        ANNOptions.initWeight = 0.1;
        ANNOptions.isRandomWeight = false;
        ANNOptions.learningRate = 0.1;
        ANNOptions.momentum = 0.1;
        ANNOptions.maxEpoch = 100;
        ANNOptions.numHiddenNeuron = 2;
        ANNOptions.threshold = 0.34;
        
        Classifier mlp;
        Instances dataSet; 
        String filename = "weather.test.arff";
        
        dataSet= WekaUtil.loadDataARFF(filename);
        dataSet = WekaUtil.nominalToBinaryFilter(dataSet);
        dataSet = WekaUtil.normalizationFilter(dataSet);
        
        
        mlp = new MultiLayerPerceptron(dataSet);
        mlp.buildClassifier(dataSet);
        WekaUtil.saveModel(filename, mlp);
        //WekaUtil.percentageSplit(dataSet, mlp, 50);
        //WekaUtil.crossValidation(dataSet, mlp);
    }
}
