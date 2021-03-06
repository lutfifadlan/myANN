/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

/**
 *
 * @author Mochamad Lutfi F
 */
public class Perceptron extends Classifier{
    private int ninput; //jumlah input
    private int ninstance; //jumlah instance
    private int nclass; //jumlah kelas
    private Instances data;
    private List<Double[]> allInstanceValue;
    private List<List<Double[]>> allWeightUpdated;
    private List<List<Double[]>> inputWeight;
    private List<List<Double[]>> allDeltaWeight;
    private List<Double[]> deltaWeightFinal;
    private List<Double[]> newWeightFinal;
    private List<List<Double>> listTargetInstance;
    private List<Double> errorTarget;
    private List<List<Double>> allErrorValue; 
    private boolean isConvergen;
    private boolean isRandom;
    private int maxEpoch;
    private int activationFunction; // 0 = sign, 1 = sigmoid
    public final Double weightInitialization = 0.0;
    public final Double treshold = 0.01;
    public final Double learningRate = 0.1;
    public final Double momentum = 0.2;
    // CONSTRUCTOR
    public Perceptron() throws IOException{
        activationFunction = 0;
        allInstanceValue = new ArrayList<>();
        listTargetInstance = new ArrayList<>();
        inputWeight = new ArrayList<>();
        allDeltaWeight = new ArrayList<>();
        deltaWeightFinal = new ArrayList<>();
        newWeightFinal = new ArrayList<>();
        allWeightUpdated = new ArrayList<>();
        allErrorValue = new ArrayList<>();
        isConvergen = false;
        errorTarget = new ArrayList<>();
        isRandom = false;
        nclass = 1;
        maxEpoch = 10;
        loadARFF("iris.arff");
    }
    
    //GETTER
    public int getNinput(){
        return ninput;
    }
    
    public int getNInstance(){
        return ninstance;
    }
    
    public Instances getData(){
        return data;
    }
    
    public List<List<Double[]>> getAllWeightUpdated(){
        return allWeightUpdated;
    }
    
    public List<List<Double[]>> getAllWeight(){
        return inputWeight;
    }
    
    public List<Double[]> getAllInstance(){
        return allInstanceValue;
    }
    
    public List<List<Double[]>> getAllDeltaWeight(){
        return allDeltaWeight;
    }
    
    public List<List<Double>> getListTarget(){
        return listTargetInstance;
    }
    
    public boolean getIsConvergen(){
        return isConvergen;
    }
    
    public final double getTreshold(){
        return treshold;
    }

    public List<List<Double>> getAllErrorValue(){
        return allErrorValue;
    } 
    
    //SETTER
    public void setNinput(int _ninput){
        ninput = _ninput;
    } 
    
    public void setAllWeightUpdated(List<List<Double[]>> _listWeightUpdated){
        allWeightUpdated= _listWeightUpdated;
    }
    
    public void setAllWeight(List<List<Double[]>> _allWeight){
        inputWeight = _allWeight;
    }
   
    public void setAllInstanceValue(List<Double[]> _allInstance){
        allInstanceValue = _allInstance;
    }
    
    public void setAllDeltaWeight(List<List<Double[]>> _allDeltaWeight){
        allDeltaWeight = _allDeltaWeight;
    }
    
    public void setTargetInstance(List<List<Double>> _listTarget){
        listTargetInstance = _listTarget;
    }

    public void setIsConvergen(boolean _isConvergen){
        isConvergen = _isConvergen;
    }

    public void setAllErrorValue(List<List<Double>> _allErrorValue){
        allErrorValue = _allErrorValue;
    }
    
    //FUNCTION
    public Double getClassIndex(int indexInstance){
        Double classIndex = 0.0;
        for(int i=0; i<nclass; i++){
            List<Double> listTarget = listTargetInstance.get(i);
            if(listTarget.get(indexInstance) == 1)
                classIndex = (double) i;
        }
        return classIndex;
    }
    
    public int highestOutputIndex(List<Double> output){
        int index = 0;
        Double maxOut = output.get(0);
        for(int i=1;i<output.size();i++){
            if(output.get(i) > maxOut){
                index = i;
                maxOut = output.get(i);
            }
        }
        return index;
    }
    
    public void initializeDeltaWeightFinal(){
        deltaWeightFinal.clear();
        for(int i=0; i<nclass; i++){
            Double[] _deltaWeightFinal = new Double[ninput - 1];
            for(int j=0; j<ninput - 1; j++){
                _deltaWeightFinal[j] = 0.0;
            }
            deltaWeightFinal.add(_deltaWeightFinal);
        }
    }
    
    public void InputWeight(boolean isRandom){
        for(int i=0;i<nclass;i++){
            List<Double[]> weightClass = new ArrayList<>();
            for(int j=0; j<ninstance;j++){
                Double[] weightInstance = new Double[ninput-1];
                for(int k=0;k<ninput-1;k++){
                    if(isRandom){
                        Random random = new Random();
                        weightInstance[k] = (double) random.nextInt(1);
                    }
                    else{
                        weightInstance[k] = 0.0;
                    }   
                }
                 weightClass.add(weightInstance);
            }
            inputWeight.add(weightClass);
        }
    }
    
    
    public void InputInstance(Instances instances){
        data = instances;
        ninput = instances.numAttributes();
        ninstance = instances.numInstances();
        System.out.println("ninput = " + ninput);
        System.out.println("ninstance = " + ninstance);
        for(int i=0; i<ninstance; i++)
        {
            Instance thisInstance = instances.instance(i);
            Double[] listInput = new Double[ninput];
            for(int j=0; j<ninput; j++){
                listInput[j] = thisInstance.value(j);
            }
            allInstanceValue.add(listInput);           
        }
    } 
    
    public void InputTargetInstances(Instances instances) {
        int numInstance = instances.numInstances();
        nclass = instances.numClasses();
        for (int i=0;i<nclass;i++) {
            List<Double> listTargetClass = new ArrayList<>();
            for (int j=0;j<numInstance;j++) {
                if (instances.instance(j).classValue() == (double) i) {
                    listTargetClass.add(1.0);
                } else {
                    listTargetClass.add(0.0);
                }
            }
            listTargetInstance.add(listTargetClass);
        }
    }
    
    public void initializeInputWeight(int outputIndex){
        List<Double[]> listInputWeight = new ArrayList<>();
        for(int i=0; i<ninstance; i++){
            Double[] _inputWeight = new Double[ninput - 1];
            for(int j=0; j<ninput-1; j++){
                _inputWeight[j] = newWeightFinal.get(outputIndex)[j];
            }
            listInputWeight.add(_inputWeight);
        }
        inputWeight.add(outputIndex, listInputWeight);
    }
    
    public void initializeNewWeightFinal(){
        newWeightFinal.clear();
        for(int i=0; i<nclass;i++){
            Double[] _newWeightFinal = new Double[ninput-1];
            for(int j=0;j<ninput-1;j++){
                _newWeightFinal[j] = 0.0;
            }
            newWeightFinal.add(_newWeightFinal);
        }
    }
    
    public double ComputeOutput(Double[] _inputInstance, Double[] _inputWeight){
        double net = weightInitialization;
        double output=0;
        for(int i=0; i<ninput-1; i++){
            net = net + (_inputInstance[i] * _inputWeight[i]);
        } 
        if(activationFunction == 0){
            if(net > 0){
                output = 1;
            }
            else output = -1;
        }
        if(activationFunction == 1){
            output = 1 / (1 - Math.exp(-1*net));
        }
        return output;
    }
    
    public double ComputeErrorTarget(Double targetOutput, Double output){
        return (targetOutput-output);
    }
    
    public double ComputeErrorEpoch(List<Double> error){
        double Errortemp, sumErrortemp;
        sumErrortemp = 0;
        for (int i=0; i<ninstance; i++){
                Errortemp = Math.pow((error.get(i)), 2);
                sumErrortemp += Errortemp;
        }
        return (0.5 * sumErrortemp);
    }
     
    public Double[] ComputeDeltaWeight(Double[] _inputInstance, double _errorTarget, int numInstance, int numOutput){
       // System.out.println("ninput 3 = "+ninput);
        Double[] deltaWeight = new Double[ninput - 1];
        for(int i=0;i<ninput-1;i++){
            double prevDeltaWeight;
            if(numInstance > 0){
                prevDeltaWeight = allDeltaWeight.get(numOutput).get(numInstance - 1)[i];
            }
            else{
                prevDeltaWeight = deltaWeightFinal.get(numOutput)[i];
            }
            deltaWeight[i] = learningRate * _inputInstance[i] * _errorTarget + momentum * prevDeltaWeight;
            if (deltaWeight[i] == -0.0)
                deltaWeight[i] = 0.0;
        }
        return deltaWeight;
    }
    
    public Double[] ComputeNewWeight(Double[] _inputWeight, Double[] deltaWeight){
        Double[] newWeight = new Double[ninput-1];
        for(int i=0;i<ninput-1;i++){
            newWeight[i] = deltaWeight[i] + _inputWeight[i];
        }
        return newWeight;
    }
    
    public void buildClassifier(Instances _instances){
        Instances instances;
        instances = WekaUtil.nominalToBinaryFilter(_instances);
        instances = WekaUtil.normalizationFilter(_instances);
        InputInstance(instances);
        InputTargetInstances(instances);
        InputWeight(isRandom);
        initializeDeltaWeightFinal();
        initializeNewWeightFinal();
        for(int i=0;i<maxEpoch;i++){
            errorTarget.clear();
            inputWeight.clear();
            allDeltaWeight.clear();
            allWeightUpdated.clear();
            for(int idx=0;idx<nclass;idx++){
                allDeltaWeight.add(new ArrayList<>());
                allWeightUpdated.add(new ArrayList<> ());
            }
            for(int j=0;j<ninstance;j++){
                for(int k=0;k<nclass;k++){
                    initializeInputWeight(k);
                    //System.out.println(allInstanceValue.get(j)[0] + inputWeight.get(k).get(j)[0]);
                    double tempOutput = ComputeOutput(allInstanceValue.get(j), inputWeight.get(k).get(j));
                    double tempErrorTarget = ComputeErrorTarget(listTargetInstance.get(k).get(j), tempOutput);
                    Double[] deltaWeight = ComputeDeltaWeight(allInstanceValue.get(j),tempErrorTarget,j,k);
                    allDeltaWeight.get(k).add(j, deltaWeight);
                    deltaWeightFinal.set(k, deltaWeight);
                    Double[] newWeight = ComputeNewWeight(inputWeight.get(k).get(j), deltaWeight);
                    allWeightUpdated.get(k).add(j,newWeight);
                    newWeightFinal.set(k, newWeight);
                 } 
            }
            for(int j=0;j<ninstance;j++){
                List<Double> listOutput = new ArrayList<>();
                for(int k=0;k<nclass;k++){
                    Double outputFinal = ComputeOutput(allInstanceValue.get(j), newWeightFinal.get(k));
                    listOutput.add(outputFinal);
                }
                int hoi = highestOutputIndex(listOutput);
                Collections.sort(listOutput);
                Double finalOutput = listOutput.get(nclass - 1);
                Double finalError = ComputeErrorTarget(listTargetInstance.get(hoi).get(j),finalOutput);
                errorTarget.add(finalError);
            }
            double mse = ComputeErrorEpoch(errorTarget);
            if(mse < treshold){
                isConvergen = true;
                break;
            }
        }
     }

   public double classifyInstance(Instance instance){
       System.out.println("ninput = " + ninput);
       Double[] input = new Double[ninput - 1];
       for(int i=0; i<instance.numAttributes()-1;i++){
           input[i] = instance.value(i);
       }
       List<Double> allOutput = new ArrayList<>();
       for(Double[] newWeight : newWeightFinal){
           Double outputNode = ComputeOutput(input,newWeight);
           allOutput.add(outputNode);
       }
       for(Double output : allOutput){
           System.out.println("Output: " + output);
       }
       int indexClass = 0;
       Double maxOutput = allOutput.get(0);
       for(int i=1;i<allOutput.size();i++){
           if(allOutput.get(i) > maxOutput){
               indexClass = i;
               maxOutput = allOutput.get(i);
           }
       }
       return indexClass;
   }
   
   public void loadARFF(String filename) throws FileNotFoundException, IOException{
       FileReader file = null;
       file = new FileReader(filename);
       BufferedReader reader = new BufferedReader(file);
       data = new Instances(reader);
       data.setClassIndex(data.numAttributes() - 1);
   }
}

