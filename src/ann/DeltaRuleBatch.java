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
        bias = 1;
        biasWeight = 0;
        inputList = new ArrayList<>();
        inputWeightList = new ArrayList<>();
        output = new ArrayList<>();
        target = new ArrayList<>();
        error = new ArrayList<>();
        deltaWeight = new ArrayList<>();
        newWeight = new ArrayList<>();
        sigmaDeltaWeight = new ArrayList<>();
        maxEpoch = 100;
        isConvergent = false;
        threshold = 0.3;
        isRandomWeight = false;
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
            for(int j=0;j<numAttributes;j++){
                if(j==0){//bias
                    input[j] = 0.0;
                }else{
                    input[j] = dataSet.instance(i).value(j-1);
                }
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
                    inputWeight = new Double[numAttributes];
                    for(int k=0;k<numAttributes;k++){
                        if(j==0){//biasWeight;
                            inputWeight[j] = biasWeight;
                        }else{
                            randomValue = new Random();
                            inputWeight[j] = (double)randomValue.nextInt(1);
                        }
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
                    inputWeight = new Double[numAttributes];
                    for(int k=0;k<numAttributes;k++){
                        if(j==0){//biasWeight;
                        inputWeight[j] = biasWeight;
                        }else{
                        inputWeight[j] = givenWeightValue;
                        }
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
        
    }
    
    public Double[] calculateDeltaWeight(Double[] inputValue, Double errorValue, 
            int neuronOutputIndex, int attrIndex){
        Double[] deltaWeightThisInstance = new Double[numAttributes-1];
        for (int k=0;k<numAttributes-1;k++) {
            double previousDeltaWeightThisAttribute;
            if (attrIndex > 0) {
                previousDeltaWeightThisAttribute = deltaWeight.get(neuronOutputIndex).get(attrIndex)[k];
            } else {
                previousDeltaWeightThisAttribute = sigmaDeltaWeight.get(neuronOutputIndex)[k];
            }
            deltaWeightThisInstance[k] = (learningRate * inputValue[k] * errorValue) 
                                         + (momentum * previousDeltaWeightThisAttribute);
        }
        return deltaWeightThisInstance;
    }
    
    public Double[] calculateNewWeightInstance(Double[] inputWeight, Double[] deltaWeight){
        Double[] newWeightThisInstance = new Double[numAttributes];
        for (int k=0;k<numAttributes;k++) {
            newWeightThisInstance[k] = inputWeight[k] + deltaWeight[k];
        }
        return newWeightThisInstance;
    }
    
    public void calculateSigmaDeltaWeight(){
        Double[] sigmaDeltaWeightPerClass;
        Double sigmaDeltaWeightPerAttribute;
        Double[] finalNewWeightPerClass;
        
        for(int i=0;i<numClasses;i++){
            sigmaDeltaWeightPerClass = new Double[numAttributes];
            for(int j=0;j<numAttributes;j++){
                sigmaDeltaWeightPerAttribute = 0.0;
                for(int k=0;k<numInstances;k++){
                    sigmaDeltaWeightPerAttribute += deltaWeight.get(i).get(k)[j];
                }
                sigmaDeltaWeightPerClass[j] = sigmaDeltaWeightPerAttribute;
            }
            sigmaDeltaWeight.add(sigmaDeltaWeightPerClass);
            //finalNewWeightPerClass = new Double[numAttributes];
            finalNewWeightPerClass = calculateNewWeightInstance(inputWeightList.get(i).get(numInstances-1), sigmaDeltaWeightPerClass);
            finalNewWeight.add(finalNewWeightPerClass);
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
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        initDataSet(instances);
        initInputValue();
        initWeightValue();
        initTargetValue();
        
        for(int i=0;i<maxEpoch;i++){
            resetData();
            //proses penghitungan output, error, deltaWeight tiap instances
            for(int j=0;j<numClasses;j++){
                for(int k=0; k<numInstances;k++){
                    //hitung output
                    double outputPerInstance = 0;
                    for(int x=0;x<numAttributes;x++) {
                        outputPerInstance += inputList.get(k)[x] * inputWeightList.get(j).get(k)[x];
                    }
                    //output.add(outputPerInstance);
                    //hitung error
                    double errorPerInstance;
                    errorPerInstance = target.get(j).get(k) - outputPerInstance;
                    //hitung deltaW = learningRate*nilai atribut* error
                    Double[] deltaWeightPerInstance = calculateDeltaWeight(inputList.get(k),errorPerInstance,j,k);
                    deltaWeight.get(j).add(k, deltaWeightPerInstance);
                    Double[] newWeightPerInstance = calculateNewWeightInstance(inputWeightList.get(j).get(k), deltaWeightPerInstance);
                    newWeight.get(k).add(j,newWeightPerInstance);
                }
            }
            // hitung sigma delta weight
            calculateSigmaDeltaWeight();
            // Add error akhir
            for (int j=0;j<numInstances;j++) {
                output = new ArrayList<>();
                for (int k=0;k<numClasses;k++) {
                    double outputFinalThisClass = 0;//computeOutputInstance(inputValue.get(j),finalNewWeight.get(k));
                    for(int x=0;x<numAttributes;x++) {
                        outputFinalThisClass += inputList.get(j)[x] * finalNewWeight.get(k)[x];
                    }
                    output.add(outputFinalThisClass);
                }
                int indexClassHighestOutput = getIndexClassHighestOutput(output);
                Collections.sort(output);
                Double finalOutputThisInstance = output.get(numClasses-1);
                Double errorThisInstance = target.get(indexClassHighestOutput).get(j) - finalOutputThisInstance;
                error.add(errorThisInstance);
            }
            // Hitung MSE Epoch
            double mseValue = 0.0;
            for (int j=0;j<numAttributes;j++) {
                mseValue += Math.pow(error.get(j), 2);
            }
            mseValue *= 0.5;
            //System.out.println("Error epoch " + (i+1) + " : " + mseValue);
            if (mseValue<threshold) {
                isConvergent = true;
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
            for(int x=0;x<numAttributes;x++) {
                outputThisNeuron += inputValue[x] * weightValue[x];
            }
            outputPerNeuron.add(outputThisNeuron);
        }
        return getIndexClassHighestOutput(outputPerNeuron);
    }
    
    
}
