
import ann.ANNOptions;
import ann.DeltaRuleBatch;
import ann.WekaUtil;
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
        ANNOptions.initWeight = 0.05;
        ANNOptions.isRandomWeight = false;
        ANNOptions.learningRate = 0.1;
        ANNOptions.momentum = 0.1;
        ANNOptions.maxEpoch = 1000;
        //ANNOptions.numHiddenNeuron = 2;
        ANNOptions.threshold = 0.34;
        
        Classifier classifier;
        Instances dataSet;
        String filename = "weather.nominal.arff";
        String filenameUnlabeled = "weather.unlabeled.nominal.arff";
        dataSet= WekaUtil.loadDataARFF(filename);
        dataSet = WekaUtil.nominalToBinaryFilter(dataSet);
        dataSet = WekaUtil.normalizationFilter(dataSet);
        
        
        //classifier = new MultiLayerPerceptron(dataSet);
        classifier = new DeltaRuleBatch();
        
        classifier.buildClassifier(dataSet);
        WekaUtil.saveModel(filename, classifier);
        //WekaUtil.percentageSplit(dataSet, classifier, 50);
        WekaUtil.crossValidation(dataSet, classifier);
        //classifier = WekaUtil.loadModel(filename);
        //WekaUtil.classify(filenameUnlabeled, classifier);
    }
}
