/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.Serializable;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 *
 * @author Fahmi
 */
public class Neuron implements Serializable{
    private static final long serialVersionUID = 0;
    
    public SortedMap<Integer,Double> input;
    public double output;
    public double error;
    public SortedMap<Integer,Double> target;
    
    public Neuron(){
        
    }
    
    public Neuron(Neuron n){
        input = n.input;
        output = n.output;
        error = n.error;
        target = n.target;
    }
    
    public Neuron(double newOutput){
        output = newOutput;
        input = new TreeMap<>();
        target = new TreeMap<>();
    }
    
    public Neuron(SortedMap<Integer, Double> newInput, double newOutput){
        input = newInput;
        output = newOutput;
    }
}
