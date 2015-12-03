/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class DeltaRuleBatch extends Classifier{
    double bias;
    double biasWeight;
    double givenWeightValue;
    double learningRate;
    double momentum;
    ArrayList<Double[]> inputList;
    ArrayList<ArrayList<Double[]>> inputWeightList;
    ArrayList<ArrayList<Double>> target;
    ArrayList<Double> output;
    ArrayList<Double> error; //target-output
    ArrayList<ArrayList<Double[]>> deltaWeight;
    ArrayList<ArrayList<Double[]>> newWeight;
    ArrayList<Double[]> sigmaDeltaWeight;
    ArrayList<Double[]> finalNewWeight;
    double errorEpoch;
    
    int maxEpoch;
    double threshold;
    boolean isRandomWeight;
    boolean isConvergent;
    
    Instances dataSet;
    int numInstances;
    int numAttributes;
    int numClasses;
    
    public DeltaRuleBatch(){
        bias = ANNOptions.bias;
        biasWeight = ANNOptions.biasWeight;
        givenWeightValue = ANNOptions.initWeight;
        learningRate = ANNOptions.learningRate;
        momentum = ANNOptions.momentum;
        maxEpoch = ANNOptions.maxEpoch;
        threshold = ANNOptions.threshold;
        isRandomWeight = ANNOptions.isRandomWeight;
        
        inputList = new ArrayList<>();
        inputWeightList = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        error = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
        sigmaDeltaWeight = new ArrayList<>();
        finalNewWeight = new ArrayList<>();
        //isConvergent = false;
        dataSet = null;
    }
    
    public DeltaRuleBatch(Instances dataSet){
        bias = ANNOptions.bias;
        biasWeight = ANNOptions.biasWeight;
        givenWeightValue = ANNOptions.initWeight;
        learningRate = ANNOptions.learningRate;
        momentum = ANNOptions.momentum;
        maxEpoch = ANNOptions.maxEpoch;
        threshold = ANNOptions.threshold;
        isRandomWeight = ANNOptions.isRandomWeight;
        
        inputList = new ArrayList<>();
        inputWeightList = new ArrayList<>();
        target = new ArrayList<>();
        output = new ArrayList<>();
        error = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
        sigmaDeltaWeight = new ArrayList<>();
        finalNewWeight = new ArrayList<>();
        //isConvergent = false;
        this.dataSet = dataSet;
    }
    
    public void setDataSet(Instances newDataSet){
        dataSet = newDataSet;
    }
    public Instances getDataSet(){
        return dataSet;
    }
    
    public void setBias(double newBias){
        bias = newBias;
    }
    public double getBias(){
        return bias;
    }
    
    public void setBiasWeight(double newBiasWeight){
        biasWeight = newBiasWeight;
    }
    public double getBiasWeight(){
        return biasWeight;
    }
    
    public void setGivenWeight(double newGivenWeight){
        givenWeightValue = newGivenWeight;
    }
    public double getGivenWeight(){
        return givenWeightValue;
    }
    
    public void setLearningRate(double newLearningRate){
        learningRate = newLearningRate;
    }
    public double getLearningRate(){
        return learningRate;
    }
    public void setMomentum(double newMomentum){
        momentum = newMomentum;
    }
    public double getMomentum(){
        return momentum;
    }
    public void setMaxEpoch(int newMaxEpoch){
        maxEpoch = newMaxEpoch;
    }
    public int getMaxEpoch(){
        return maxEpoch;
    }
    public void setThreshold(double newThreshold){
        threshold = newThreshold;
    }
    public double getThreshold(){
        return threshold;
    }
    
    @Override
    public Capabilities getCapabilities(){
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        return result;
    }
    
    public void initDataSet(Instances i){
        dataSet = i;
        numInstances = i.numInstances();
        numAttributes = i.numAttributes();
        numClasses = i.numClasses();
    }
    public void initInputValue(){
        Double[] input;
        for(int i=0;i<numInstances;i++){
            input = new Double[numAttributes];
            for(int j=0;j<numAttributes-1;j++){
                //if(j==0){//bias
                //    input[j] = 0.0;
                //}else{
                    input[j] = dataSet.instance(i).value(j);
                //}
            }
            inputList.add(input);
        }
    }
    
    public void initWeightValue(){
        ArrayList inputWeightPerClass;
        Double[] inputWeight;
        if(isRandomWeight){
            Random randomValue;
            for(int i=0;i<numClasses;i++){
                inputWeightPerClass = new ArrayList<>();
                for(int j=0;j<numInstances;j++){
                    inputWeight = new Double[numAttributes-1];
                    for(int k=0;k<numAttributes-1;k++){
                        //if(j==0){//biasWeight;
                        //    inputWeight[k] = biasWeight;
                        //}else{
                            randomValue = new Random();
                            inputWeight[k] = randomValue.nextDouble() - 0.05;
                        //}
                    }
                    inputWeightPerClass.add(inputWeight);
                }
                inputWeightList.add(inputWeightPerClass);
            }
        }
        else{//givenWeight
            for(int i=0;i<numClasses;i++){
                inputWeightPerClass = new ArrayList<>();
                for(int j=0;j<numInstances;j++){
                    inputWeight = new Double[numAttributes-1];
                    for(int k=0;k<numAttributes-1;k++){
                        //if(j==0){//biasWeight;
                        //inputWeight[k] = biasWeight;
                        //}else{
                        inputWeight[k] = givenWeightValue;
                        //}
                    }
                    inputWeightPerClass.add(inputWeight);
                }
                inputWeightList.add(inputWeightPerClass);
            }
        }
    }
    
    public void initTargetValue(){
        ArrayList<Double> targetPerInstance;
        for (int i=0;i<numClasses;i++) {
            targetPerInstance = new ArrayList<>();
            for (int j=0;j<numInstances;j++) {
                if(dataSet.instance(j).classValue() == (double) i) {
                    targetPerInstance.add(1.0);
                }else{
                    targetPerInstance.add(0.0);
                }
            }
            target.add(targetPerInstance);
        }
    }
    
    public void resetData(){
        output.clear();
        error.clear();
        inputWeightList.clear();
        deltaWeight.clear();
        newWeight.clear();
        for (int i=0;i<numClasses;i++) {
            deltaWeight.add(new ArrayList<>());
            newWeight.add(new ArrayList<>());
        }
    }
    
    public Double[] calculateDeltaWeight(Double[] inputValue, Double errorValue, 
            int neuronOutputIndex, int instanceIndex){
        Double[] deltaWeightThisInstance = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            double previousDeltaWeightThisAttribute;
            if (instanceIndex > 0) {
                previousDeltaWeightThisAttribute = deltaWeight.get(neuronOutputIndex).get(instanceIndex-1)[k];
            } else {
                previousDeltaWeightThisAttribute = sigmaDeltaWeight.get(neuronOutputIndex)[k];
            }
            deltaWeightThisInstance[k] = (learningRate * inputValue[k] * errorValue) 
                                         + (momentum * previousDeltaWeightThisAttribute);
        }
        return deltaWeightThisInstance;
    }
    
    public Double[] calculateNewWeight(Double[] inputWeight, Double[] deltaWeight){
        Double[] newWeightThisInstance = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            newWeightThisInstance[k] = inputWeight[k] + deltaWeight[k];
        }
        return newWeightThisInstance;
    }
    
    public void calculateSigmaDeltaWeight(){
        Double[] sigmaDeltaWeightPerClass;
        Double sigmaDeltaWeightPerAttribute;
        Double[] finalNewWeightPerClass;
        
        for(int i=0;i<numClasses;i++){
            sigmaDeltaWeightPerClass = new Double[numAttributes-1];
            for(int j=0;j<numAttributes-1;j++){
                sigmaDeltaWeightPerAttribute = 0.0;
                for(int k=0;k<numInstances;k++){
                    sigmaDeltaWeightPerAttribute += deltaWeight.get(i).get(k)[j];
                }
                sigmaDeltaWeightPerClass[j] = sigmaDeltaWeightPerAttribute;
            }
            sigmaDeltaWeight.set(i,sigmaDeltaWeightPerClass);
            //finalNewWeightPerClass = new Double[numAttributes];
            finalNewWeightPerClass = calculateNewWeight(inputWeightList.get(i).get(numInstances-1), sigmaDeltaWeightPerClass);
            finalNewWeight.set(i, finalNewWeightPerClass);
        }
    }
    
    public int getIndexClassHighestOutput(ArrayList<Double> outputPerNeuron){
        int indexClassMaxOutput = 0;
        Double maxOutput = outputPerNeuron.get(0);
        for (int i=1;i<outputPerNeuron.size();i++) {
            if (outputPerNeuron.get(i) > maxOutput) {
                indexClassMaxOutput = i;
                maxOutput = outputPerNeuron.get(i);
            }
        }
        return indexClassMaxOutput;
    }
    
    public void initInputWeightPerEpoch(){
        for(int i=0;i<numClasses;i++){
            ArrayList<Double[]> inputWeightPerClass = new ArrayList<>();
            for(int j=0;j<numInstances;j++){
                Double[] inputWeightPerInstance = new Double[numAttributes-1];
                System.arraycopy(finalNewWeight.get(i), 0, inputWeightPerInstance, 0, numAttributes-1);
                inputWeightPerClass.add(inputWeightPerInstance);
            }
            inputWeightList.add(i,inputWeightPerClass);
        }
    }
    
    public void initFinalNewWeight() {
        finalNewWeight.clear();
        Double[] finalNewWeightPerClass;
        for (int i=0;i<numClasses;i++) {
            finalNewWeightPerClass = new Double[numAttributes-1];
            for (int j=0;j<numAttributes-1;j++) {
                finalNewWeightPerClass[j] = 0.0;
            }
            finalNewWeight.add(i, finalNewWeightPerClass);
        }
    }
    
    public void initSigmaDeltaWeight() {
        sigmaDeltaWeight.clear();
        Double[] sigmaDeltaWeightPerClass;
        for (int i=0;i<numClasses;i++) {
            sigmaDeltaWeightPerClass = new Double[numAttributes-1];
            for (int j=0; j<numAttributes-1; j++) {
                sigmaDeltaWeightPerClass[j] = 0.0;
            }
            sigmaDeltaWeight.add(i, sigmaDeltaWeightPerClass);
        }
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        instances.deleteWithMissingClass();
        
        initDataSet(instances);
        initInputValue();
        initWeightValue();
        initTargetValue();
        initFinalNewWeight();
        initSigmaDeltaWeight();
        
        for(int i=0;i<maxEpoch;i++){
            resetData();
            initInputWeightPerEpoch();
            
            //proses penghitungan output, error, deltaWeight tiap instances
            for(int j=0;j<numClasses;j++){
                for(int k=0; k<numInstances;k++){
                    //hitung output
                    double outputPerInstance = 0;
                    for(int x=0;x<numAttributes-1;x++) {
                        outputPerInstance += inputList.get(k)[x] * inputWeightList.get(j).get(k)[x];
                    }
                    //output.add(outputPerInstance);
                    //hitung error
                    double errorPerInstance;
                    errorPerInstance = target.get(j).get(k) - outputPerInstance;
                    //hitung deltaW = learningRate*nilai atribut* error
                    Double[] deltaWeightPerInstance = calculateDeltaWeight(inputList.get(k),errorPerInstance,j,k);
                    deltaWeight.get(j).add(k,deltaWeightPerInstance);
                    Double[] newWeightPerInstance = calculateNewWeight(inputWeightList.get(j).get(k), deltaWeightPerInstance);
                    newWeight.get(j).add(k,newWeightPerInstance);
                }
            }
            // hitung sigma delta weight
            calculateSigmaDeltaWeight();
            // Add error akhir
            for (int j=0;j<numInstances;j++) {
                output = new ArrayList<>();
                for (int k=0;k<numClasses;k++) {
                    double outputFinalThisClass = 0;//computeOutputInstance(inputValue.get(j),finalNewWeight.get(k));
                    for(int x=0;x<numAttributes-1;x++) {
                        outputFinalThisClass += inputList.get(j)[x] * finalNewWeight.get(k)[x];
                    }
                    output.add(outputFinalThisClass);
                }
                int indexClassHighestOutput = getIndexClassHighestOutput(output);
                Collections.sort(output);
                Double finalOutputThisInstance = output.get(numClasses-1);
                Double errorThisInstance = target.get(indexClassHighestOutput).get(j) - finalOutputThisInstance;
                error.add(errorThisInstance);
                //System.out.println(errorThisInstance);
            }
            // Hitung MSE Epoch
            double mseValue = 0.0;
            for (int j=0;j<numAttributes;j++) {
                mseValue += Math.pow(error.get(j), 2);
            }
            mseValue *= 0.5;
            System.out.println("Error epoch " + (i+1) + " : " + mseValue);
            if (mseValue<threshold) {
                //isConvergent = true;
                break;
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance instance){
        Double[] inputValue = new Double[numAttributes-1];
        for (int i=0;i<instance.numAttributes()-1;i++) {
            inputValue[i] = instance.value(i);
        }
        // Hitung output setiap neuron, cari yang terbesar
        ArrayList<Double> outputPerNeuron = new ArrayList<>();
        for (Double[] weightValue : finalNewWeight) {
            double outputThisNeuron = 0;
            for(int x=0;x<numAttributes-1;x++) {
                outputThisNeuron += inputValue[x] * weightValue[x];
                System.out.println(outputThisNeuron);
            }
            outputPerNeuron.add(outputThisNeuron);
        }
        return getIndexClassHighestOutput(outputPerNeuron);
    }    
}
