/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.SortedMap;
import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Fahmi
 */
public class MultiLayerPerceptron extends Classifier{
    
    private double bias;
    private double biasWeight;
    private double initWeight;
    private double isRandomWeight;
    private double learningRate;
    private double momentum;
    private int numEpoch;
    private int numInputNeuron;
    private int numHiddenNeuron;
    private int numOutputNeuron;
    private double threshold;
    
    private Instances dataSet;
    private ArrayList<Double> errorList;
    private Map<Integer, Double> classMap;
    private Map<String, ArrayList<Integer>> neuronType;
    private Map<Integer, Neuron> neurons;
    private Map<Integer, Map<Integer, Double[]>> weights;

    public MultiLayerPerceptron(Instances instances){
        dataSet = instances;
        
        if (ANNOptions.isRandomWeight) {
            System.out.println("random Weight");
            double rangeMin = -0.05;
            double rangeMax = 0.05;
            Random r = new Random();
            initWeight = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        }else{
            System.out.println("given Weight");
            initWeight = ANNOptions.initWeight;
        }
        
        numInputNeuron = dataSet.numAttributes();
        numHiddenNeuron = ANNOptions.numHiddenNeuron;
        numOutputNeuron = dataSet.numClasses();
        momentum = ANNOptions.momentum;
        learningRate = ANNOptions.learningRate;
        numEpoch = ANNOptions.maxEpoch;
        threshold = ANNOptions.threshold;

        neurons = new HashMap<>();
        weights = new HashMap<>();
        neuronType = new HashMap<>();
        
        bias = ANNOptions.bias;
        initBiasValue();
        biasWeight = ANNOptions.biasWeight;
        initInputNeurons();
        initHiddenNeuron(numHiddenNeuron);
    }
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        return result;
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
        getCapabilities().testWithFail(instances);
        //instances = new Instances(instances);
        instances.deleteWithMissingClass();
        dataSet = new Instances(instances);
        
        initOutputNeuron(dataSet.numClasses());
        //setNumNeuron(instances.numClasses(), false); //output
        setNeuronConnectivity();

        double temp, max = 0; 
        int maxOutputNeuron = -1;
        ArrayList<Integer> hiddenNeurons = neuronType.get("hidden");
        ArrayList<Integer> outputNeurons = neuronType.get("output");

        //for (int epoch = 0; epoch < numEpoch; epoch++) {
        int epoch = 0;
        boolean underThreshold = false;
        do{
            errorList = new ArrayList<>();
            for (int i = 0; i < dataSet.numInstances(); i++) {
                setInstanceTarget(i);
                //feed-forward
                for (Integer hidNeuron : hiddenNeurons) {
                    computeOutputValue(hidNeuron, i);
                }
                for (Integer hidNeuron : outputNeurons) {
                    computeOutputValue(hidNeuron, i);
                }
                //backPropagation
                computeOutputNeuronError(i);
                computeHiddenNeuronError(i);
                updateWeights(i);

                //feed-forward again
                for (Integer hidNeuron : hiddenNeurons) {
                    computeOutputValue(hidNeuron, i);
                }
                for (Integer anOutput : outputNeurons) {
                    temp = computeOutputValue(anOutput, i);
                    if (temp > max) {
                        max = anOutput;
                        maxOutputNeuron = anOutput;
                    }
                }
                addError(i, neurons.get(maxOutputNeuron).target.get(i) - neurons.get(maxOutputNeuron).input.get(i));
            }

            if (Double.compare(calculateMSE(), threshold) < 0) {
                underThreshold = true;
            }
        }while((epoch < numEpoch)&&(!underThreshold));
        
        //printNeuron();
        //printWeight();
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        ArrayList<Integer> hiddenNeurons = neuronType.get("hidden");
        ArrayList<Integer> outputNeurons = neuronType.get("output");
        ArrayList<Integer> neuronBefore;
        double weight, maxValue = 0;
        int classIndex = 0;

        for (Integer hidNeuron : hiddenNeurons) {
            neurons.put(hidNeuron, new Neuron());
        }
        for (Integer outNeuron : outputNeurons) {
            neurons.put(outNeuron, new Neuron());
        }

        for (Integer hidNeuron : hiddenNeurons) {
            weight = 0;
            neuronBefore = findNodeBefore(hidNeuron);

            for (int j=0; j<neuronBefore.size(); j++) {
                if(j==0){ //bias
                    weight += weights.get(hidNeuron).get(j)[0];
                }else{
                    weight += weights.get(hidNeuron).get(j)[0] * instance.value(j - 1);
                }
            }
            neurons.get(hidNeuron).input = new TreeMap<>();
            neurons.get(hidNeuron).input.put(0, sigmoid(weight));
        }

        for (Integer outNeuron : outputNeurons) {
            weight = 0;
            neuronBefore = findNodeBefore(outNeuron);

            for (int j=0; j < neuronBefore.size(); j++) {
                if(j==0) { //bias
                    weight += weights.get(outNeuron).get(j)[0];
                }else{
                    weight += weights.get(outNeuron).get(neuronBefore.get(j))[0] * neurons.get(neuronBefore.get(j)).input.get(0);
                }
            }
            neurons.get(outNeuron).input = new TreeMap<>();
            neurons.get(outNeuron).input.put(0, sigmoid(weight));
        }

        for (Integer outNeuron : outputNeurons) {
            if (Double.compare(neurons.get(outNeuron).input.get(0), maxValue) > 0) {
                maxValue = neurons.get(outNeuron).input.get(0);
                classIndex = outNeuron;
            }
        }
        return classMap.get(classIndex);
    }
    
    private void initBiasValue(){
        SortedMap<Integer,Double> tempBias = new TreeMap<>();
        for (int i=0; i< dataSet.numInstances(); i++){
            tempBias.put(i, bias);
        }
        neurons.put(0, new Neuron(tempBias, 0.0));
        //System.out.println(neurons.toString());
    }
    private void initInputNeurons(){
        Map<Integer, SortedMap<Integer,Double>> attrDataSet = initDataSet();
        
        for(int i=1; i<=dataSet.numAttributes(); i++){
            neurons.put(i, new Neuron(attrDataSet.get(i-1), 0));
            //System.out.println(attrDataSet.get(i));
        }
        //System.out.println(neurons.toString());
    }

    private Map<Integer, SortedMap<Integer, Double>> initDataSet() {
        Map<Integer, SortedMap<Integer, Double>> tempDataSet = new HashMap<>();
        SortedMap<Integer, Double> attrData;
        for (int i=0; i<dataSet.numAttributes(); i++) {
            attrData = new TreeMap<>();
            for (int j=0; j<dataSet.numInstances(); j++) {
                attrData.put(j, dataSet.instance(j).value(i));
            }
            tempDataSet.put(i, attrData);
        }
        return tempDataSet;
    }
    
    public void setNeurons(int numNeuron, boolean isHidden){
        String type;
        Map<Integer, Double[]> temp;
        ArrayList<Integer> neuronNumber = new ArrayList<>();
        Double[] d = new Double[3];
        d[0] = biasWeight;
        d[1] = 0.0;
        d[2] = 0.0;

        if(isHidden){
            type = "hidden";
            //numHiddenNeuron = numNeuron;
        }
        else{
            type = "output";
            //numOutputNeuron = numNeuron;
        }
        int counter = neurons.size();
        //System.out.println(neurons.size());
        for (int i = 0; i < numNeuron; i++) {
            if (weights.containsKey(counter+i)){
                temp = weights.get(counter+i);
            }else{
                temp = new HashMap<>();
            }
            temp.put(0, d);
            neurons.put(counter+i, new Neuron(0)); //neuron with output = 0
            weights.put(counter+i, temp);
            neuronNumber.add(counter+i);
        }
        neuronType.put(type, neuronNumber);
        //System.out.println(weights.toString());
        //System.out.println(neurons.toString());
    }
    private void initHiddenNeuron(int numNeuron){
        setNeurons(numNeuron, true);
    }
    private void initOutputNeuron(int numNeuron){
        setNeurons(numNeuron, false);
    }
    
    public void setNeuronConnectivity() {
        Double[] d = new Double[3];
        d[0] = initWeight; // new weight
        d[1] = 0.0; // old weight
        d[2] = 0.0; // delta weight
        Map<Integer, Double[]> temp;
        
        // input neuron to hidden neuron
        for (int i=1; i<=numInputNeuron; i++) {
            for (int j=numInputNeuron+1; j<=(numInputNeuron+numHiddenNeuron); j++) {
                if(weights.containsKey(j)){
                    temp = weights.get(j);
                }else{
                    temp = new HashMap<>();
                }
                temp.put(i, d);
                weights.put(j, temp);
            }
        }
        //System.out.println("Input ke hidden");
        //System.out.println(weights.toString());
        // hidden to output
        for (int i=numInputNeuron+1; i<=(numInputNeuron+numHiddenNeuron); i++) {
            for (int j=(numInputNeuron+numHiddenNeuron)+1; j<=(numInputNeuron+numHiddenNeuron+numOutputNeuron); j++) {
                if (weights.containsKey(j)) {
                    temp = weights.get(j);
                } else {
                    temp = new HashMap<>();
                }
                temp.put(i, d);
                weights.put(j, temp);
            }
        }
        //System.out.println("hidden ke output");
        //System.out.println(weights.toString());
        
        initClassMap();
    }
    
    public void initClassMap() {
        classMap = new HashMap<>();
        ArrayList<Integer> output = neuronType.get("output");
        //System.out.println("initClassMap");
        //System.out.println(output);
        int numClass = dataSet.classAttribute().numValues();
        //System.out.println(numClass);
        for (int i=0; i<numClass; i++) {
            classMap.put(output.get(i), (double) i);
        }
        //System.out.println(classMap.toString());
    }
    
    
    public ArrayList<Integer> findNodeBefore(int node) {
        return new ArrayList<>(weights.get(node).keySet());
    }

    public ArrayList<Integer> findNodeAfter(int node) {
        ArrayList<Integer> nodeAfterList = new ArrayList<>();

        weights.entrySet().stream().forEach((weight) -> {
            weight.getValue().entrySet().stream().filter(refs -> 
                    refs.getKey() == node).forEach(refs -> nodeAfterList.add(weight.getKey()));
        });
        return nodeAfterList;
    }

    public double findWeight(int node1, int node2) {
        return weights.get(node2).get(node1)[0];
    }
    
    public double computeOutputValue(int node, int instanceNo) {
        double output = 0;
        ArrayList<Integer> refs = findNodeBefore(node);
        ArrayList<Integer> refsAfter = findNodeAfter(node);

        for (Integer ref : refs) {
            output += findWeight(ref, node) * neurons.get(ref).input.get(instanceNo);
        }

        Neuron n = new Neuron(neurons.get(node));
        n.output = output;

        for (int i = 0; i < dataSet.numInstances(); i++) {
            n.input.put(i, sigmoid(output));
        }
        neurons.put(node, n);

        for (Integer ref : refsAfter) {
            n = new Neuron(neurons.get(ref));
            n.input.put(0, sigmoid(output));
            neurons.put(ref, n);
        }

        return output;
    }
    
    public void setInstanceTarget(int instanceNo) {
        ArrayList<Integer> outputNeurons = neuronType.get("output");
        Neuron n;
        for (int i = 0; i < outputNeurons.size(); i++) {
            n = neurons.get(outputNeurons.get(i));
            if (Double.compare((double) i, dataSet.instance(instanceNo).classValue()) == 0) {
                n.target.put(instanceNo, 1.0);
            }else{
                n.target.put(instanceNo, 0.0);
            }
            neurons.put(outputNeurons.get(i), n);
        }
    }

    public void computeOutputNeuronError(int instanceNo) {
        ArrayList<Integer> outputNeurons = neuronType.get("output");
        Neuron n;
        for (Integer outNeuron : outputNeurons) {
            n = neurons.get(outNeuron);
            //error = output * (1-output) *(target - output)
            n.error = n.input.get(0) * (1 - n.input.get(0)) * (n.target.get(instanceNo) - n.input.get(0));
            neurons.put(outNeuron, n);
        }
    }
    
    public void computeHiddenNeuronError(int instanceNo) {
        ArrayList<Integer> hiddenNeurons = neuronType.get("hidden");
        ArrayList<Integer> nodeAfterList;
        double temp;

        Neuron n;
        for (Integer hidNeuron : hiddenNeurons) {
            n = neurons.get(hidNeuron);
            //error = output*(1-output)
            n.error = n.input.get(instanceNo) * (1 - n.input.get(instanceNo));

            nodeAfterList = findNodeAfter(hidNeuron);
            temp = 0;
            for (Integer outputNeuron : nodeAfterList) {
                temp += (neurons.get(outputNeuron).error * findWeight(hidNeuron, outputNeuron));
            }
            n.error *= temp;
            neurons.put(hidNeuron, n);
        }
    }
    
    public void addError(int instanceNo, double diffTargetOutput) {
        errorList.add(instanceNo, diffTargetOutput);
    }

    public double calculateMSE() {
        //equation: 0.5 * (sum (t-o)^2)
        double sumError = 0;
        for (Double e : errorList) {
            sumError += Math.pow(e, 2);
        }
        return sumError/2;
    }
    
    public void updateWeights(int instanceNo) {
        Double[] tempDouble;
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight : weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                tempDouble = new Double[3];
                double d = realWeight.getValue()[0];
                tempDouble[2] = learningRate *
                        neurons.get(realWeight.getKey()).input.get(instanceNo) *
                        neurons.get(weight.getKey()).error + (momentum * realWeight.getValue()[2]); // compute delta weight
                tempDouble[1] = realWeight.getValue()[2]; // hold the old delta weight
                tempDouble[0] = d + tempDouble[2];
                realWeight.setValue(tempDouble);
            }
        }
    }
    
    /*public void printNeuron() {
        System.out.println("Neuron - Input Value (Activation) - Target - Net Value - Error");
        for (Map.Entry<Integer, Neuron> neuron : neurons.entrySet()) {
            System.out.print(neuron.getKey() + " ");
            System.out.print(neuron.getValue().input);
            System.out.print(" "+ neuron.getValue().target + " ");
            System.out.print(" [" + neuron.getValue().output + "] ");
            System.out.println(" [" + neuron.getValue().error + "] ");
        }
    }

    public void printWeight() {
        System.out.println();
        System.out.println("Connection - New Weight - Old Weight - Delta Weight");
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: weights.entrySet()) {
            for (Map.Entry<Integer, Double[]> realWeight : weight.getValue().entrySet()) {
                System.out.print(realWeight.getKey() + "-" + weight.getKey() + " ");
                System.out.print(realWeight.getValue()[0] + " ");
                System.out.print(realWeight.getValue()[1] + " ");
                System.out.println(realWeight.getValue()[2] + " ");
            }
        }
    }*/
    
    public double sigmoid(double value){
        double exp = 1 + Math.exp(-value);
        if(exp != 0){
            return (1/exp);
        }else{
            return 0;
        }
    }
}
