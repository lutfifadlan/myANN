/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;

import java.util.*;

/**
 *
 * @author Mochamad Lutfi F
 */
public class Perceptron {
    private int ninput; //jumlah input
    private int ninstance; //jumlah instance
    private double[] instance; //array nilai input
    private List<Double> allInstance;
   // private int[] numinput; //penomoran input
    private int maxNode; //nomor node paling besar
    private int currentNode; //nomer node saat ini 
    private double[] listWeight;
    //private double[] listWeightUpdated;
    private List<Double> allWeightUpdated;
    private List<Double> allWeight;
    private double deltaWeight;
    private double[] listDeltaWeight;
    private List<Double> allDeltaWeight;
    private double target;
    private double[] listTarget;
    private double output;
    private double[] listOutput;
    private List<Double> allOutput;
    private List<Double> allNetFunction;
    private Random random;
    private boolean isConvergen;
    private int nepoch;
    private double[] errorPerInstance; //target-output
    private double errorValue;
    private List<Double> allErrorValue; 
    public final double weightInitialization = 0;
    public final double treshold = 0.01;
    public final double learningRate = 0.1;
    // CONSTRUCTOR
    public Perceptron(int _ninput, int _ninstance){
        ninstance = _ninstance;
        ninput = _ninput;
        instance = new double[ninput];
        listWeight = new double[ninput];
        listDeltaWeight = new double[ninput];
        listTarget = new double[ninstance];
        errorPerInstance = new double [ninstance];
        allInstance = new ArrayList<>();
        allWeight = new ArrayList<>();
        allDeltaWeight = new ArrayList<>();
        allWeightUpdated = new ArrayList<>();
        allOutput = new ArrayList<>();
        allNetFunction = new ArrayList<>();
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
    
    public List<Double> getAllWeightUpdated(){
        return allWeightUpdated;
    }
    
    public List<Double> getAllWeight(){
        return allWeight;
    }
    
    public double[] getInstance(){
        return instance;
    }
    
    public List<Double> getAllInstance(){
        return allInstance;
    }
    
    public double getDeltaWeight(){
        return deltaWeight;
    }
    
    public double[] getListDeltaWeight(){
        return listDeltaWeight;
    }
    
    public List<Double> getAllDeltaWeight(){
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
    
    public double[] getListOutput(){
        return listOutput;
    }
    
    public List<Double> getAllOutput(){
        return allOutput;
    }
    
    public List<Double> getAllNetFunction(){
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
    
    public List<Double> getAllErrorValue(){
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
    
    public void setAllWeightUpdated(List<Double> _listWeightUpdated){
        allWeightUpdated= _listWeightUpdated;
    }
    
    public void setAllWeight(List<Double> _allWeight){
        allWeight = _allWeight;
    }
    
    public void setInstance(double[] _instance){
        instance = _instance;
    }
    
    public void setAllInstance(List<Double> _allInstance){
        allInstance = _allInstance;
    }
    
    public void setDeltaWeight(double _deltaWeight){
        deltaWeight = _deltaWeight;
    }
    
    public void setListDeltaWeight(double[] _listDeltaWeight){
        listDeltaWeight = _listDeltaWeight;
    }
    
    public void setAllDeltaWeight(List<Double> _allDeltaWeight){
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
    
    public void setListOutput(double[] _listOutput){
        listOutput = _listOutput;
    }
    
    public void setAllOutput(List<Double> _allOutput){
        allOutput = _allOutput;
    }
    
    public void setAllNetFunction(List<Double> _allNetFunction){
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
    
    public void setAllErrorValue(List<Double> _allErrorValue){
        allErrorValue = _allErrorValue;
    }
    
    //FUNCTION
     public void PerceptronLearning(double[] _instance, double[] _listWeight){
        input();
        SignFunction(ComputeNetFunction());
       // ComputeErrorPerInstance();
        ComputeErrorEpoch();
        
        /*
        if(isConvergen){
            System.out.println("Network sudah konvergen");
        }
        else{
            PerceptronLearning();
        }*/
    }
    
    public void input(){
        for(int i=0; i<ninput; i++)
        {
            allWeight.add(listWeight[i]);
            allInstance.add(instance[i]);
            //System.out.println("allWeight " + "[" + i + "]" + "=" + allWeight.get(i));
        }
        currentNode = ninput + 1;
    } 
    
    public void addInput(double[] input){
        for(int i=0; i<ninput; i++)
            allInstance.add(input[i]);
    }
    
    public double ComputeNetFunction(){
        double net;
        net = weightInitialization;
        for(int i=0; i<ninput; i++){
            net = net + listWeight[i]*instance[i];
            allNetFunction.add(net);
        } 
        return net;
    }
    
    public void SignFunction(double netOutput){
        for(int i=0;i<ninstance;i++){
            if (allNetFunction.get(i) >= 0){
                output = 1;
                allOutput.add(output);
            }
            else{
                output = -1;       
                allOutput.add(output);
            }            
        }
    }
   
    public double ComputeErrorPerInstance(int indexInstance){
            return listTarget[indexInstance] - allOutput.get(indexInstance);
    }
    
    public void ComputeErrorEpoch(){
        double Errortemp, sumErrortemp;
        sumErrortemp = 0;
        for (int i=0; i<ninstance; i++){
            {
               // System.out.println("listTarget= " + listTarget[i]);
             //   System.out.println("allOutput= " + allOutput.get(i));
                Errortemp = 0.5 * Math.pow((listTarget[i] - allOutput.get(i)), 2);
             //   System.out.println("Errortemp= " + Errortemp);
                sumErrortemp += Errortemp;
            }
        }
        errorValue = sumErrortemp;
      //  System.out.println("errorValue = " + errorValue);
        allErrorValue.add(errorValue);
    }
     
    public void EpochStatus(int indexEpoch){
        if(allErrorValue.get(indexEpoch) < treshold){
            isConvergen = true;
        }
        else{
            nepoch++;
            ComputeArrayofDeltaWeight(listTarget[0] ,allOutput.get(0));
            ComputeNewWeight();
        }
    }
     
    public double ComputeDeltaWeight(double _target, double _output, double _input){
     //   System.out.println("Masuk ComputeDeltaWeight");
        double temp = learningRate * (_target - _output) * _input;
        if((_target - _output) == 0||_input == 0)
            temp = 0;
        return temp;
    }
    
    public void ComputeArrayofDeltaWeight(double _target, double _output){
      //  System.out.println("masuk");
        for (int i=0; i<ninput; i++){
            listDeltaWeight[i] = ComputeDeltaWeight(_target,_output, instance[i]);
            allDeltaWeight.add(listDeltaWeight[i]);
        //    System.out.println("allDeltaWeight ["+ i + "]" + " = " + allDeltaWeight.get(i));
        }
    }
    
    public void ComputeNewWeight(){
        for(int i=0; i<ninput; i++){
         //   System.out.println("listWeight ["+ i + "]" + " = " + listWeight[i]);
        //    System.out.println("allDeltaWeight ["+ i + "]" + " = " + allDeltaWeight.get(i));
            listWeight[i] = listWeight[i] + allDeltaWeight.get(i);
            allWeightUpdated.add(listWeight[i]);
        }
    }
    
    public void printPerEpoch(){
        for(int i=0; i<ninput; i++)
            System.out.println("Input ke-" + (i+1) + " = " + instance[i]);
        for(int i=0; i<ninput; i++)
            System.out.println("Weight ke-" + (i+1) + " = " + allWeight.get(i));
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
}

