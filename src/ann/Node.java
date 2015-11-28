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
public class Node {
    private int ninput; //jumlah input
    private double[] listinput; //array nilai input
   // private int[] numinput; //penomoran input
    private int maxnode; //nomor node paling besar
    private int currentnode; //nomer node saat ini
   // private double weight;
    private final double weightInitialization;
    private double[] listWeight;
    private double[] listWeightupdated;
    private List<Double> allWeight;
    private double deltaWeight;
    private double[] listDeltaWeight;
    private List<Double> allDeltaWeight;
    private double learningRate;
    private double target;
    private double[] listTarget;
    private double output;
    private double[] listOutput;
    private List<Double> allOutput;
    private List<Double> allNetFunction;
    private Random random;
    private boolean isConvergen;
    private final double treshold;
    private int nepoch;
    private double errorValue;
    private List<Double> allErrorValue; 
    
    public Node(int ninput){
        listWeight = new double[ninput];
        listDeltaWeight = new double[ninput];
        weightInitialization = 0;
        currentnode = 0;
        isConvergen = false;
        treshold = 0.01;
        allWeight.add(listWeight[0]);
        nepoch = 0;
        errorValue = 0;
        allErrorValue.add(errorValue);
    }
        
    public int getNinput(){
        return ninput;
    }
    
    public double[] getListInput(){
        return listinput;
    }
    
    private int getMaxnode() //nomor node paling besar
    private int getCurrentnode() //nomer node saat ini
   // private double weight;
    private final double getWeightInitialization;
    private double[] listWeight;
    private double[] listWeightupdated;
    private List<Double> allWeight;
    private double deltaWeight;
    private double[] listDeltaWeight;
    private List<Double> allDeltaWeight;
    private double learningRate;
    private double target;
    private double[] listTarget;
    private double output;
    private double[] listOutput;
    private List<Double> allOutput;
    private List<Double> allNetFunction;
    private Random random;
    private boolean isConvergen;
    private final double treshold;
    private int nepoch;
    private double errorValue;
    private List<Double> allErrorValue; 
    
    
    
    public void inputWeight(double[] _listWeight, int _ninput){
        listWeight = _listWeight;
        for(int i=0; i<_ninput; i++)
            allWeight.add(listWeight[i]);
        currentnode = _ninput + 1;
        maxnode = currentnode + (_ninput - 1);
    }
    
    public void SignFunction(double[] _listinput){
        double NetFunctionTreshold = 0;
        for(int i=0;i<ninput;i++){
            if (NetFunction(_listinput)[i] > NetFunctionTreshold){
                output = 1;
                allOutput.add(output);
            }
            else{
                output = -1;       
                allOutput.add(output);
            }            
        }
    }
    
    public void PerceptronLearning(double[] inputweight, int _ninput){
        inputWeight(inputweight,_ninput);
        SignFunction(NetFunction(inputweight));
        ComputeErrorEpoch(_ninput);
        EpochStatus(ninput, inputweight);
        if(isConvergen){
            System.out.println("Network sudah konvergen");
        }
        
        else{
            PerceptronLearning(listWeightupdated, _ninput);
        }
    }
    
    public double[] NetFunction(double[] _inputweight){
        double[] net = new double[0];
        for(int i=0; i<ninput; i++){
            net[i] = weightInitialization + listWeight[i]*_inputweight[i];
            allNetFunction.add(net[i]);
        } 
        return net;
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
            listWeightupdated[i] = oldweight[i]; 
            allWeight.add(oldweight[i]);
        }
        newweight = oldweight;
        return newweight;
    }
    
    public double ComputeErrorEpoch(int _ninput){
        double Errortemp, sumErrortemp;
        sumErrortemp = 0;
        for (int i=0; i<_ninput; i++){
            for (int j=currentnode; j<maxnode;j++){
                Errortemp = ((1/2) * Math.pow((listTarget[i] - allOutput.get(j)), 2));
                sumErrortemp += Errortemp;
            }
        }
        errorValue = sumErrortemp;
        allErrorValue.add(errorValue);
        return errorValue;
    }
    
    public void EpochStatus(int _ninput, double[] oldweight){
        if(ComputeErrorEpoch(_ninput) < treshold){
            isConvergen = true;
        }
        else{
            nepoch++;
            ComputeNewWeight(_ninput, oldweight, listDeltaWeight);
        }
    }
    
    public void printPerEpoch(){
        for(int i=0; i<maxnode; i++)
        {
            for(int j=0; j<ninput;j++)
            {
                for(int k=0; k<nepoch; k++){
                    System.out.println("Weight ke-" + (i+1) + " = " + allWeight.get(i));
                    System.out.println("Input ke-" + j + " = " + listinput[j]);
                    System.out.println("Error Epoch ke-" + k + " = " + allErrorValue.get(k));
                }
            }
        }
    }
    
}

