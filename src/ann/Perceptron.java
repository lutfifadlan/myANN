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
    private double[] listWeightUpdated;
    private List<Double> allWeight;
    private double deltaWeight;
    private double[] listDeltaWeight;
    private List<Double> allDeltaWeight;
    private double learningRate;
    private double target;
    private double[] listTarget;
    private double output;
    private double[] listOutput;
    private double netFunction;
    private List<Double> allOutput;
    private List<Double> allNetFunction;
    private Random random;
    private boolean isConvergen;
    private int nepoch;
    private double[] errorPerInstance; //target-output
    private double errorValue;
    private List<Double> allErrorValue; 
    public final double weightInitialization;
    public final double treshold;
    
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
        allOutput = new ArrayList<Double>();
        allNetFunction = new ArrayList<>();
        allErrorValue = new ArrayList<>();
        weightInitialization = 0;
        currentNode = 0;
        maxNode = ninput-1;
        isConvergen = false;
        treshold = 0.01;
        nepoch = 1;
        errorValue = 0;
    }
     
    //GETTER
    public int getNinput(){
        return ninput;
    }
    
    public double[] getListInput(){
        return instance;
    }
    
    public int getMaxnode() {
        return maxNode;
    }
    public int getCurrentnode() //nomer node saat ini
    {
        return currentNode;
    }
    
    public final double getWeightInitialization(){
        return weightInitialization;
    }
         
    public double[] getListWeight(){
        return listWeight;
    }
    public double[] getListWeightupdated(){
        return listWeightUpdated;
    }
    public List<Double> getAllWeight(){
        return allWeight;
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
    public double getLearningRate(){
        return learningRate;
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
    public void setListInput(double[] _instance){
        instance = _instance;
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
    
    public void setListWeightupdated(double[] _listWeightUpdated){
        listWeightUpdated= _listWeightUpdated;
    }
    
    public void setAllWeight(List<Double> _allWeight){
        allWeight = _allWeight;
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
    
    public void setLearningRate(double _learningRate){
        learningRate = _learningRate;
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
     public void PerceptronLearning(double[] inputweight){
        input();
        SignFunction(ComputeNetFunction(inputweight));
        ComputeErrorPerInstance();
        ComputeErrorEpoch();
        EpochStatus(inputweight);
        if(isConvergen){
            System.out.println("Network sudah konvergen");
        }
        else{
            PerceptronLearning(listWeightUpdated);
        }
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
    
    public double ComputeNetFunction(double[] _inputweight){
        double net = 0;
        for(int i=0; i<ninput; i++){
            net = net + weightInitialization + listWeight[i]*_inputweight[i];
            allNetFunction.add(net);
        } 
        netFunction = net;
        return net;
    }
    
    public void SignFunction(double netOutput){
        double NetFunctionTreshold = 0;
        boolean success = false;
        for(int i=0;i<ninstance;i++){
            if (netFunction > NetFunctionTreshold){
                output = 1;
                allOutput.add(output);
            }
            else{
                output = -1;       
                allOutput.add(output);
            }            
        }
    }
   
    public double[] ComputeErrorPerInstance(){
        double error[] = new double[ninstance];
        for(int i=0;i<ninstance;i++){
            error[i] = listTarget[i] - allOutput.get(i);
        }
        return error;
    }
    
    public double ComputeErrorEpoch(){
        double Errortemp, sumErrortemp;
        sumErrortemp = 0;
        for (int i=0; i<ninstance; i++){
            {
                Errortemp = ((1/2) * Math.pow((listTarget[i] - allOutput.get(i)), 2));
                sumErrortemp += Errortemp;
            }
        }
        errorValue = sumErrortemp;
        allErrorValue.add(errorValue);
        return errorValue;
    }
     
    public void EpochStatus(double[] oldweight){
        if(ComputeErrorEpoch() < treshold){
            isConvergen = true;
        }
        else{
            nepoch++;
            ComputeNewWeight(ninput, oldweight, listDeltaWeight);
        }
    }
     
    public double ComputeDeltaWeight(double _learningRate, double _target, double _output, double _input){
        return (_learningRate * (_target - _output) * _input);
    }
    
    public double[] ArrayofDeltaWeight(int _ninput, double _learningRate, double _target, double _output, double _input){
        double[] _ArrofDelW;
        _ArrofDelW = new double[_ninput];
        for (int i=0; i<_ninput; i++){
            listDeltaWeight[i] = ComputeDeltaWeight(_learningRate,_target,_output,_input);
            allDeltaWeight.add(listDeltaWeight[i]);    
        }
        return _ArrofDelW;
    }
    
    public double[] ComputeNewWeight(int _ninput, double[] oldweight, double[] _deltaweight){
        double[] newweight;
        for(int i=0; i<ninput; i++){
            oldweight[i] = oldweight[i] + _deltaweight[i];
            listWeightUpdated[i] = oldweight[i]; 
            allWeight.add(oldweight[i]);
        }
        newweight = oldweight;
        return newweight;
    }
    
    public void printPerEpoch(){
        System.out.println("Masuk");
        for(int i=0; i<ninput; i++)
        {
            System.out.println("Weight ke-" + (i+1) + " = " + allWeight.get(i));
            System.out.println("Input ke-" + (i+1) + " = " + instance[i]);
        }
        for(int i=0; i<ninstance; i++){
            System.out.println("Error instance ke-" + (i+1) + " = " + errorPerInstance[i]);
        }
        for(int k=0; k<nepoch; k++)
            System.out.println("Error Epoch ke-" + k + " = " + allErrorValue.get(k));
    }
}

