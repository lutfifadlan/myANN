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
public class Perceptron {
    private int ninput; //jumlah input
    private int ninstance; //jumlah instance
    private int nclass; //jumlah kelas
    private double[] instance; //array nilai input
    private Instances data;
    private List<Double[]> allInstanceValue;
   // private int[] numinput; //penomoran input
    private int maxNode; //nomor node paling besar
    private int currentNode; //nomer node saat ini 
    private double[] listWeight;
    //private double[] listWeightUpdated;
    private List<List<Double[]>> allWeightUpdated;
    private List<List<Double[]>> inputWeight;
    private double deltaWeight;
    private double[] listDeltaWeight;
    private List<List<Double[]>> allDeltaWeight;
    private List<Double[]> deltaWeightFinal;
    private List<Double[]> newWeightFinal;
    private List<List<Double>> targetInstance;
    private double target;
    private double[] listTarget;
    private double output;
    private List<Double> listOutput;
    private List<Double[]> allOutput;
    private List<List<Double>> allNetFunction;
    private Random random;
    private boolean isConvergen;
    private int nepoch;
    private double[] errorPerInstance; //target-output
    private double errorValue;
    private List<Double> errorTarget;
    private List<Double> listErrorValue;
    private List<List<Double>> allErrorValue; 
    private boolean isRandom;
    private int maxEpoch;
    private int activationFunction; // 0 = sign, 1 = sigmoid
    public final double weightInitialization = 0;
    public final double treshold = 0.01;
    public final double learningRate = 0.1;
    public final double momentum = 0.2;
    // CONSTRUCTOR
    public Perceptron() throws IOException{
        readARFF("C:\\Program Files\\Weka-3-6\\data\\weather.numeric.arff");
        listOutput = new ArrayList<>();
        allInstanceValue = new ArrayList<>();
        targetInstance = new ArrayList<>();
        inputWeight = new ArrayList<>();
        allDeltaWeight = new ArrayList<>();
        deltaWeightFinal = new ArrayList<>();
        newWeightFinal = new ArrayList<>();
        allWeightUpdated = new ArrayList<>();
        allOutput = new ArrayList<>();
        allNetFunction = new ArrayList<>();
        listErrorValue = new ArrayList<>();
        allErrorValue = new ArrayList<>();
        currentNode = 0;
        maxNode = ninput-1;
        isConvergen = false;
        nepoch = 1;
        errorValue = 0;
        isRandom = false;
        maxEpoch = 10;
    }
    
    public Perceptron(int _ninput, int _ninstance){
        ninstance = _ninstance;
        ninput = _ninput;
        instance = new double[ninput];
        listWeight = new double[ninput];
        listDeltaWeight = new double[ninput];
        listTarget = new double[ninstance];
        errorPerInstance = new double [ninstance];
        listOutput = new ArrayList<>();
        allInstanceValue = new ArrayList<>();
        inputWeight = new ArrayList<>();
        allDeltaWeight = new ArrayList<>();
        allWeightUpdated = new ArrayList<>();
        allOutput = new ArrayList<>();
        allNetFunction = new ArrayList<>();
        listErrorValue = new ArrayList<>();
        allErrorValue = new ArrayList<>();
        currentNode = 0;
        maxNode = ninput-1;
        isConvergen = false;
        nepoch = 1;
        errorValue = 0;
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
    
    public double[] getInstance(){
        return instance;
    } 
    
    public int getMaxnode() {
        return maxNode;
    }
    public int getCurrentnode() //nomer node saat ini
    {
        return currentNode;
    }
         
    public double[] getListWeight(){
        return listWeight;
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
    
    public double getDeltaWeight(){
        return deltaWeight;
    }
    
    public double[] getListDeltaWeight(){
        return listDeltaWeight;
    }
    
    public List<List<Double[]>> getAllDeltaWeight(){
        return allDeltaWeight;
    }
    
    public double getTarget(){
        return target;
    }
    
    public double[] getListTarget(){
        return listTarget;
    }
    
    public double getOutput(){
        return output;
    }
    
    public List<Double> getListOutput(){
        return listOutput;
    }
    
    public List<Double[]> getAllOutput(){
        return allOutput;
    }
    
    public List<List<Double>> getAllNetFunction(){
        return allNetFunction;
    }
    
    public boolean getIsConvergen(){
        return isConvergen;
    }
    
    public final double getTreshold(){
        return treshold;
    }
    
    public int getNepoch(){
        return nepoch;
    }
    
    public double getErrorValue(){
        return errorValue;
    }
    
    public List<List<Double>> getAllErrorValue(){
        return allErrorValue;
    } 
    
    //SETTER
    public void setNinput(int _ninput){
        ninput = _ninput;
    } 
   
    public void setMaxNode(int _maxNode){
        maxNode = _maxNode;
    }
    
    public void setCurrentNode(int _currentNode){
        currentNode = _currentNode;
    }
    
    public void setListWeight(double[] _listWeight){
        listWeight = _listWeight;
    }
    
    public void setAllWeightUpdated(List<List<Double[]>> _listWeightUpdated){
        allWeightUpdated= _listWeightUpdated;
    }
    
    public void setAllWeight(List<List<Double[]>> _allWeight){
        inputWeight = _allWeight;
    }
    
    public void setInstance(double[] _instance){
        instance = _instance;
    }
    
    public void setAllInstanceValue(List<Double[]> _allInstance){
        allInstanceValue = _allInstance;
    }
    
    public void setDeltaWeight(double _deltaWeight){
        deltaWeight = _deltaWeight;
    }
    
    public void setListDeltaWeight(double[] _listDeltaWeight){
        listDeltaWeight = _listDeltaWeight;
    }
    
    public void setAllDeltaWeight(List<List<Double[]>> _allDeltaWeight){
        allDeltaWeight = _allDeltaWeight;
    }
    
    public void setTarget(double _target){
        target = _target;
    }
    
    public void setListTarget(double[] _listTarget){
        listTarget = _listTarget;
    }
    
    public void setOutput(double _output){
        output = _output;
    }
    
    public void setListOutput(List<Double> _listOutput){
        listOutput = _listOutput;
    }
    
    public void setAllOutput(List<Double[]> _allOutput){
        allOutput = _allOutput;
    }
    
    public void setAllNetFunction(List<List<Double>> _allNetFunction){
        allNetFunction = _allNetFunction;
    }
    
    public void setIsConvergen(boolean _isConvergen){
        isConvergen = _isConvergen;
    }
   
    public void setNepoch(int _nepoch){
        nepoch  = _nepoch;
    }
    
    public void setErrorValue(double _errorValue){
        errorValue = _errorValue;
    }
    
    public void setAllErrorValue(List<List<Double>> _allErrorValue){
        allErrorValue = _allErrorValue;
    }
    
    //FUNCTION
    
          
        /*
        if(isConvergen){
            System.out.println("Network sudah konvergen");
        }
        else{
            PerceptronLearning();
        }*/
    
    public void setNominalToBinary() throws Exception{
        NominalToBinary NTB = new NominalToBinary();
        NTB.setInputFormat(data);
        data = new Instances(Filter.useFilter(data, NTB));
    }
    
    public Double getClassIndex(int indexInstance){
        Double classIndex = 0.0;
        for(int i=0; i<nclass; i++){
            List<Double> listTarget = targetInstance.get(i);
            if(listTarget.get(indexInstance) == 1)
                classIndex = (double) i;
        }
        return classIndex;
    }
    
    public void InputInstance(Instances instances){
        ninput = instances.numAttributes();
        ninstance = instances.numInstances();
        for(int i=0; i<ninstance; i++)
        {
            Instance thisInstance = instances.instance(i);
            Double[] listInput = new Double[ninstance];
            for(int j=0; j<ninput; j++){
                listInput[j] = thisInstance.value(j);
            }
            allInstanceValue.add(listInput);
            //inputWeight.add(listWeight[i]);
            //allInstance.add(instance[i]);
            //System.out.println("inputWeight " + "[" + i + "]" + "=" + inputWeight.get(i));
        }
       // currentNode = ninput + 1;
    } 
    
    public void intializeDeltaWeightFinal(){
        for(int i=0; i<nclass; i++){
            Double[] _deltaWeightFinal = new Double[ninstance - 1];
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
                inputWeight.add(weightClass);
            }
            
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
            targetInstance.add(listTargetClass);
        }
    }
    
    public void initializeNewWeightFinal(){
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
        //List<Double> net = new ArrayList<>();
        //net.add(weightInitialization);
        for(int i=0; i<ninstance-1; i++){
            net = net + _inputInstance[i] * _inputWeight[i];
            //allNetFunction.add(net);
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
    
    public void SignFunction(double netOutput){
        //for(int i=0;i<ninstance;i++){
            //if (allNetFunction.get(i) >= 0){
            if(netOutput >= 0){
                output = 1;
                listOutput.add(output);
            }
            else{
                output = -1;       
                listOutput.add(output);
            }            
    }
   
    /*public double ComputeErrorPerInstance(int indexInstance){
            return listTarget[indexInstance] - allOutput.get(indexInstance);
    }*/
    
    public double ComputeErrorEpoch(List<Double> error){
        double Errortemp, sumErrortemp;
        sumErrortemp = 0;
        for (int i=0; i<ninstance; i++){
            {
               // System.out.println("listTarget= " + listTarget[i]);
             //   System.out.println("allOutput= " + allOutput.get(i));
                Errortemp = 0.5 * Math.pow((error.get(i)), 2);
             //   System.out.println("Errortemp= " + Errortemp);
                sumErrortemp += Errortemp;
            }
        }
        errorValue = sumErrortemp;
        return errorValue;
        //listErrorValue.add(errorValue);
      //  System.out.println("errorValue = " + errorValue);
        
    }
     
   /* public void EpochStatus(int indexEpoch){
        if(listErrorValue.get(indexEpoch) < treshold){
            isConvergen = true;
        }
        else{
            nepoch++;
            ComputeArrayofDeltaWeight(listTarget[0] ,listOutput.get(0));
            ComputeNewWeight(indexEpoch, 1);
        }
    }*/
     
    public Double[] ComputeDeltaWeight(Double[] _inputInstance, double errorTarget, int numInstance, int numOutput){
     //   System.out.println("Masuk ComputeDeltaWeight");
        Double[] deltaWeight = new Double[ninstance - 1];
        for(int i=0;i<ninstance-1;i++){
            double prevDeltaWeight;
            if(numInstance > 0){
                prevDeltaWeight = allDeltaWeight.get(numOutput).get(numInstance - 1)[i];
            }
            else{
                prevDeltaWeight = deltaWeightFinal.get(numOutput)[i];
            }
            deltaWeight[i] = learningRate * _inputInstance[i] * errorTarget + momentum * prevDeltaWeight;
        }
        return deltaWeight;
       /* double temp = learningRate * (_target - _output) * _input;
        if((_target - _output) == 0||_input == 0)
            temp = 0;
        return temp;*/
    }
    
   /* public double[] ComputeArrayofDeltaWeight(double _target, double _output){
      //  System.out.println("masuk");
        for (int i=0; i<ninput; i++){
            listDeltaWeight[i] = ComputeDeltaWeight(_target,_output, instance[i]);
   
        //    System.out.println("allDeltaWeight ["+ i + "]" + " = " + allDeltaWeight.get(i));
        }
       // allDeltaWeight.add(listDeltaWeight);
        return listDeltaWeight;
    }*/
    
    public Double[] ComputeNewWeight(Double[] _inputWeight, Double[] deltaWeight){
       // for(int i=0; i<ninput; i++){
         //   System.out.println("listWeight ["+ i + "]" + " = " + listWeight[i]);
        //    System.out.println("allDeltaWeight ["+ i + "]" + " = " + allDeltaWeight.get(i));
       //     inputWeight.get(CurrentEpoch).get(numInstance)[i] = inputWeight.get(CurrentEpoch).get(numInstance)[i] + listDeltaWeight[i];  
        //}
        //allWeightUpdated.add(inputWeight.get(CurrentEpoch));
        Double[] newWeight = new Double[ninstance-1];
        for(int i=0;i<ninstance-1;i++){
            newWeight[i] = deltaWeight[i] + _inputWeight[i];
        }
        return newWeight;
    }
    
    public void printPerEpoch(){
        for(int i=0; i<ninput; i++)
            System.out.println("Input ke-" + (i+1) + " = " + instance[i]);
        for(int i=0; i<ninput; i++)
            System.out.println("Weight ke-" + (i+1) + " = " + inputWeight.get(i));
        for(int i=0; i<ninstance; i++){
            System.out.println("Output instance ke-" + (i+1) + " = " + allOutput.get(i));
        }
        for(int k=0; k<nepoch; k++)
            System.out.println("Error Epoch ke-" + k + " = " + allErrorValue.get(k));
        if(nepoch > 1)
        {    
            for(int i=0; i<ninput; i++)
                System.out.println("Delta Weight ke-" + (i+1) + " = " + allDeltaWeight.get(i));
        }   
    }

    public void readARFF(String filename) throws FileNotFoundException, IOException{
        FileReader file = null;
        file = new FileReader(filename);
        BufferedReader reader = new BufferedReader(file);
        data = new Instances(reader);
        data.setClassIndex(data.numAttributes() - 1);
    }
    
    public void buildClassifier(Instances instances){
        //   readARFF("C:\\Program Files\\Weka-3-6\\data\\weather.numeric.arff");
        InputInstance(data);
        InputWeight(isRandom);
        InputTargetInstances(data);
          // SignFunction(ComputeNetFunction());
          // ComputeErrorPerInstance();
    //        ComputeErrorEpoch();
        //System.out.println(allInstanceValue.get(0)[0] + inputWeight.get(0).get(0)[0]);
        for(int i=0;i<maxEpoch;i++){
            for(int j=0;j<ninstance;j++){
                for(int k=0;k<nclass;k++){
                   // System.out.println(allInstanceValue.get(j) + inputWeight.get(k).get(j));
                    double tempOutput = ComputeOutput(allInstanceValue.get(j), inputWeight.get(k).get(j));
                    double tempErrorTarget = ComputeErrorTarget(targetInstance.get(k).get(j), tempOutput);
                    Double[] deltaWeight = ComputeDeltaWeight(allInstanceValue.get(j),tempErrorTarget,j,k);
                    allDeltaWeight.get(k).add(j,deltaWeight);
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
                Collections.sort(listOutput);
                Double finalOutput = listOutput.get(nclass - 1);
                Double finalError = ComputeErrorTarget(getClassIndex(j),finalOutput);
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
}

