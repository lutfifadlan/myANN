/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;

import java.util.*;
import weka.core.Instance;

/**
 *
 * @author Mochamad Lutfi F
 */
public class Node {
    private int ninput; //jumlah input
    private double input; //nilai suatu input
    private double[] listinput; //array nilai input
    private int[] numinput; //penomoran input
    private double weight;
    private double weightinisialisation;
    private double[] listweight;
    private List<Double> allweight;
    private double deltaweight;
    private double[] listdeltaweight;
    private List<Double> alldeltaweight;
    private double learningrate;
    private double target;
    private double[] listtarget;
    private double output;
    private double[] listoutput;
    private List<Double> alloutput;
    private List<Double> allnetfunction;
    private Random random;
    private boolean isConvergen;
    private double treshold;
    
    public Node(Random _random){
        listweight = new double[0];
        listdeltaweight = new double[0];
        random = _random;
        weightinisialisation = random.nextDouble();
        listdeltaweight[0] = 0;
        isConvergen = false;
        treshold = 0.01;
        allweight.add(listweight[0]);
    }
    
    // Instance berupa data nominal (nama) atau numerik ?
   // public 
    
    public void SignFunction(double[] _listinput){
        double NetFunctionTreshold = 0;
        for(int i=0;i<ninput;i++){
            if (NetFunction(_listinput)[i] > NetFunctionTreshold){
                output = 1;
                alloutput.add(output);
            }
            else{
                output = -1;       
                alloutput.add(output);
            }            
        }
    }
    
    public void PerceptronLearning(){
        SignFunction(NetFunction(listinput));
        
    }
    
    public double[] NetFunction(double[] _inputweight){
        double[] net = new double[0];
        for(int i=0; i<ninput; i++){
            net[i] = weightinisialisation + listweight[i]*_inputweight[i];
            allnetfunction.add(net[i]);
        } 
        return net;
    }
    
    public double ComputeDeltaWeight(double _learningrate, double _target, double _output, double _input){
        return (_learningrate * (_target - _output) * _input);
    }
    
    public double[] ArrayofDeltaWeight(int _ninput, double _learningrate, double _target, double _output, double _input){
        double[] _ArrofDelW;
        _ArrofDelW = new double[_ninput];
        for (int i=0; i<_ninput; i++){
            listdeltaweight[i] = ComputeDeltaWeight(_learningrate,_target,_output,_input);
            alldeltaweight.add(listdeltaweight[i]);    
        }
        return _ArrofDelW;
    }
    
    public double[] ComputeNewWeight(int _ninput, double[] oldweight, double[] _deltaweight){
        double[] newweight;
        for(int i=0; i<ninput; i++){
            oldweight[i] = oldweight[i] + _deltaweight[i];
            allweight.add(oldweight[i]);
        }
        newweight = oldweight;
        return newweight;
    }
    
    public double ComputeErrorEpoch(int _ninput){
        double Errortemp, sumErrortemp;
        sumErrortemp = 0;
        for (int i=0; i<_ninput; i++){
            Errortemp = ((1/2) * Math.pow((listtarget[i] - listoutput[i]), 2));
            sumErrortemp += Errortemp;
        }
        return sumErrortemp;
    }
    
    public void EpochStatus(boolean _isKonvergen, int _ninput, double[] oldweight, double[] _deltaweight){
        if(ComputeErrorEpoch(_ninput) < treshold){
            _isKonvergen = true;
        }
        else{
            ComputeNewWeight(_ninput, oldweight, _deltaweight);
        }
    }
    
}

