/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;

/**
 *
 * @author Mochamad Lutfi F
 */
public class NodeConnection {
    protected NodeConnection[] input_list; //list input yang masuk ke unit
    protected NodeConnection[] output_list; //list output yang keluar dari uniy
    protected int ninput; //jumlah node input
    protected int noutput; //jumlah node output
    protected int[] input_number; //input ke-berapa
    protected int[] output_number; //output ke-berapa
    protected double value; //value dari unit
    protected double error; //error dari unit
    protected boolean isWeightUpdated;
    protected int type; //tipe node, apakah terhubung atau tidak
    
    public NodeConnection(){
        input_list = new NodeConnection[0];
        output_list = new NodeConnection[0];
        ninput = 0;
        noutput = 0;
        input_number = new int[0];
        output_number = new int[0];
        value = 0;
        error = 0;
        isWeightUpdated = false;
        type = 0; // 0 = tidak terhubung 
    }
    // GETTER  
    public NodeConnection[] getInput(){
        return input_list;
    }
    
    public NodeConnection[] getOutput(){
        return output_list;
    }
    
    public int getNInput(){
        return ninput;
    }
    
    public int getNOutput(){
        return noutput;
    }
    
    public int[] getInputNum(){
        return input_number;
    }
    
    public int[] getOutputNum(){
        return output_number;
    }
    
    
    public double getValue(){
        return value;
    }
    
    public double getError(){
        return error;
    }
 
    public int getType(){
        return type;
    }
    
    //SETTER
    public void setInput(NodeConnection[] _input_list){
        input_list = _input_list;
    }
    
    public void setOutput(NodeConnection[] _output_list){
        output_list = _output_list;
    }
    
    public void setNinput(int _ninput){
        ninput = _ninput;
    }
    
    public void setNoutput(int _noutput){
        noutput = _noutput;
    }
    
    public void setInputNum(int[] _input_number){
        input_number = _input_number;
    } 
    
    public void setOutputNum(int[] _output_number){
        output_number = _output_number;
    }
    
    public void setValue(double _value){
        value = _value;
    }
    
    public void setError(double _error){
        error = _error;
    }
            
    public void setType(int _type){
        type = _type;
    }
    
}
