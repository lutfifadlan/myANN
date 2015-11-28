/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Fahmi
 */
public class WekaUtil {
    private static final String pathDataSet = "dataSet/";
    private static final String pathModel = "model/";
    
    public static Instances loadDataARFF(String filename){
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(pathDataSet + filename);
            Instances dataSet;
            dataSet = source.getDataSet();
            if (dataSet.classIndex() == -1){
                dataSet.setClassIndex(dataSet.numAttributes() - 1);
            }
            return dataSet;
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public static Instances loadDataCSV(String filename)
    {
        try {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(pathDataSet + filename));
            Instances dataSet = csvLoader.getDataSet();
            if(dataSet.classIndex() == -1){
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        } catch (IOException ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public static Instances removeAttribute(Instances oldData, int attIndex){
        try {
            String[] options = new String[2];
            options[0] = "-R";                                    // "range"
            options[1] = String.valueOf(attIndex);                                     // first attribute
            
            Remove remove = new Remove();                         // new instance of filter
            remove.setOptions(options);                           // set options
            remove.setInputFormat(oldData);                          // inform filter about dataset **AFTER** setting options
            Instances newData = Filter.useFilter(oldData, remove);
            return newData;
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public static Instances supervisedResample(Instances oldData, double sampleSizePercent){
        try {
            String Filteroptions="-B 1.0";
            Resample sampler = new Resample();
            sampler.setOptions(weka.core.Utils.splitOptions(Filteroptions));
            sampler.setRandomSeed((int)System.currentTimeMillis());
            sampler.setSampleSizePercent(sampleSizePercent);
            sampler.setInputFormat(oldData);
            Instances newData = Resample.useFilter(oldData,sampler);
            return newData;
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public static Instances unsupervisedResample(Instances oldData, double sampleSizePercent){
        try {
            String Filteroptions="-B 1.0";
            weka.filters.unsupervised.instance.Resample sampler = new weka.filters.unsupervised.instance.Resample();
            sampler.setOptions(weka.core.Utils.splitOptions(Filteroptions));
            sampler.setRandomSeed((int)System.currentTimeMillis());
            sampler.setSampleSizePercent(sampleSizePercent);
            sampler.setInputFormat(oldData);
            Instances newData = Resample.useFilter(oldData,sampler);
            return newData;
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public static Classifier buildClassifier(Instances dataSet, Classifier c)
    {
        try {
            c.buildClassifier(dataSet);
            return c;
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    public void testClassifier(Instances dataSet, Instances dataTest, Classifier classifier){
        if(dataSet!=null){
            try {
                // evaluate classifier and print some statistics
                Evaluation eval = new Evaluation(dataSet);
                eval.evaluateModel(classifier, dataTest);
                System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
                System.out.println(eval.toMatrixString());
            } catch (Exception ex) {
                Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
            }
        }else{
            System.out.println("Data is null");
        }
    }
    public void crossValidation(Instances dataSet, Classifier c){
        if(dataSet!=null){
            try {
                // evaluate classifier and print some statistics
                Evaluation eval = new Evaluation(dataSet);
                eval.crossValidateModel(c, dataSet, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n")); 
                System.out.println(eval.toMatrixString());
            } catch (Exception ex) {
                Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
            }
        }else{
            System.out.println("Data is null");
        }
    }
    public void percentageSplit(Instances dataSet, Classifier classifier, int percent){
            // Percent split
            int trainSize = (int) Math.round(dataSet.numInstances() * percent / 100);
            int testSize = dataSet.numInstances() - trainSize;
            Instances trainSet = new Instances(dataSet, 0, trainSize);
            Instances testSet = new Instances(dataSet, trainSize, testSize);
            // train classifier
        try {
            classifier.buildClassifier(trainSet);
            // evaluate classifier and print some statistics
            Evaluation eval = new Evaluation(trainSet);
            eval.evaluateModel(classifier, testSet);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString());
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    public static void saveModel(String filename, Classifier c){
        try {
            SerializationHelper.write(pathModel+filename, c);
            System.out.println("Model has been saved");
        } catch (Exception ex) {
            Logger.getLogger(WekaUtil.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    public Classifier loadModel(String filename) throws Exception{
        Classifier c = (Classifier) SerializationHelper.read(pathModel+filename);
        System.out.println("Model has been loaded");
        return c;
    }
    
    public void classify(String filename, Classifier classifier) throws Exception{
        // load unlabeled data and set class attribute
        Instances unlabeled = loadDataARFF(filename);
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        // create copy
        Instances labeled = new Instances(unlabeled);
        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = classifier.classifyInstance(labeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        // save newly labeled data
        ConverterUtils.DataSink.write("labeled_"+filename, labeled);

        //print hasil
        System.out.println("Classification Result");
        System.out.println("# - actual - predicted - distribution");
        for (int i = 0; i < labeled.numInstances(); i++) {
            double pred = classifier.classifyInstance(labeled.instance(i));
            double[] dist = classifier.distributionForInstance(labeled.instance(i));
            System.out.print((i+1) + " - ");
            System.out.print(labeled.instance(i).toString(labeled.classIndex()) + " - ");
            System.out.print(labeled.classAttribute().value((int) pred) + " - ");
            System.out.println(Utils.arrayToString(dist));
        }
    }
}
