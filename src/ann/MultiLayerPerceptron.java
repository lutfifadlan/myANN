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
import weka.filters.Filter;

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
    private int numHiddenNeuron;
    private int numOutputNeuron;
    private double threshold;
    
    private Instances dataSet;
    private ArrayList<Double> error;
    private Map<Integer, Double> classMap;
    private Map<String, ArrayList<Integer>> neuronType;
    private Map<Integer, Neuron> neurons;
    private Map<Integer, Map<Integer, Double[]>> weights;

    public MultiLayerPerceptron(Instances instances){
        dataSet = instances;
        
        bias = ANNOptions.bias;
        biasWeight = ANNOptions.biasWeight;
        
        if (ANNOptions.isRandomWeight) {
            double rangeMin = -0.05;
            double rangeMax = 0.05;
            Random r = new Random();
            initWeight = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        }else{
            initWeight = ANNOptions.initWeight;
        }
        
        numHiddenNeuron = ANNOptions.numHiddenNeuron;
        numOutputNeuron = dataSet.numClasses();
        momentum = ANNOptions.momentum;
        learningRate = ANNOptions.learningRate;
        numEpoch = ANNOptions.maxEpoch;
        threshold = ANNOptions.threshold;

        neurons = new HashMap<>();
        weights = new HashMap<>();
        neuronType = new HashMap<>();
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
        getCapabilities().testWithFail(instances);
        //instances = new Instances(instances);
        instances.deleteWithMissingClass();
        dataSet = new Instances(instances);
        //filterNominal
        /*nomToBinFilter.setInputFormat(dataSet);
        dataSet = Filter.useFilter(instances, nomToBinFilter);
        normalizeFilter.setInputFormat(dataSet);
        dataSet = Filter.useFilter(instances, normalizeFilter);
        */
        initInputNeurons();
        initHiddenNeuron(numHiddenNeuron);
        initOutputNeuron(dataSet.numClasses());
        //setNumNeuron(instances.numClasses(), false); //output
        setNeuronConnectivity();

        double temp, max = 0; int biggestOutNeuron = -1;
        ArrayList<Integer> hidden = neuronType.get("hidden");
        ArrayList<Integer> output = neuronType.get("output");

        for (int epoch = 0; epoch < numEpoch; epoch++) {
            error = new ArrayList<>();
            for (int i = 0; i < dataSet.numInstances(); i++) {
                setInstanceTarget(i);

                for (Integer aHidden : hidden) {
                    computeOutputValue(aHidden, i);
                }
                for (Integer anOutput : output) {
                    computeOutputValue(anOutput, i);
                }

                computeOutputNeuronError(i);
                computeHiddenNeuronError(i);

                updateWeights(i);

                for (Integer aHidden : hidden) {
                    computeOutputValue(aHidden, i);
                }
                for (Integer anOutput : output) {
                    temp = computeOutputValue(anOutput, i);
                    if (temp > max) {
                        max = anOutput;
                        biggestOutNeuron = anOutput;
                    }
                }
                addError(i, neurons.get(biggestOutNeuron).target.get(i) - neurons.get(biggestOutNeuron).input.get(i));
            }

            if (Double.compare(computeMSE(), threshold) < 0) {
                break;
            }
        }

        //printNeuron();
        //printWeight();
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        ArrayList<Integer> hiddNeuron = neuronType.get("hidden");
        ArrayList<Integer> outNeuron = neuronType.get("output");
        ArrayList<Integer> neuronBefore;
        double weight, maxValue = 0;
        int classIndex = 0;

        for (Integer hidden : hiddNeuron) {
            neurons.put(hidden, new Neuron());
        }
        for (Integer out : outNeuron) {
            neurons.put(out, new Neuron());
        }

        for (Integer aHiddNeuron : hiddNeuron) {

            weight = 0;
            neuronBefore = findNodeBefore(aHiddNeuron);

            for (int j = 0; j < neuronBefore.size(); j++) {
                if (j == 0) { //bias
                    weight += weights.get(aHiddNeuron).get(j)[0];
                } else {
                    weight += weights.get(aHiddNeuron).get(j)[0] * instance.value(j - 1);
                }
            }
            neurons.get(aHiddNeuron).input = new TreeMap<>();
            neurons.get(aHiddNeuron).input.put(0, sigmoid(weight));
        }

        for (Integer anOutNeuron : outNeuron) {

            weight = 0;
            neuronBefore = findNodeBefore(anOutNeuron);

            for (int j = 0; j < neuronBefore.size(); j++) {
                if (j == 0) { //bias
                    weight += weights.get(anOutNeuron).get(j)[0];
                } else {
                    weight += weights.get(anOutNeuron).get(neuronBefore.get(j))[0] * neurons.get(neuronBefore.get(j)).input.get(0);
                }
            }
            neurons.get(anOutNeuron).input = new TreeMap<>();
            neurons.get(anOutNeuron).input.put(0, sigmoid(weight));
        }

        for (Integer out : outNeuron) {
            if (Double.compare(neurons.get(out).input.get(0), maxValue) > 0) {
                maxValue = neurons.get(out).input.get(0);
                classIndex = out;
            }
        }
        return classMap.get(classIndex);
        
    }
    
    public void initBiasValue(){
        SortedMap<Integer,Double> tempBias = new TreeMap<>();
        for (int i=0; i< dataSet.numInstances(); i++){
            tempBias.put(i, bias);
        }
        neurons.put(0, new Neuron(tempBias, 0.0));
    }
    private void initInputNeurons(){
        Map<Integer, SortedMap<Integer,Double>> attrDataSet = initDataSet();
        
        for(int i=0; i<dataSet.numAttributes(); i++){
            neurons.put(i, new Neuron(attrDataSet.get(i), 0));
        }
    }

    private Map<Integer, SortedMap<Integer, Double>> initDataSet() {
        Map<Integer, SortedMap<Integer, Double>> tempDataSet = new HashMap<>();
        SortedMap<Integer, Double> attrData;
        for (int i=0; i<dataSet.numAttributes(); i++) {
            attrData = new TreeMap<>();
            for (int j=0; j<dataSet.numInstances(); j++) {
                attrData.put(j, dataSet.instance(j).value(i));
            }
            //why i+1?
            tempDataSet.put(i + 1, attrData);
        }
        return tempDataSet;
    }
    
    public void setNeurons(int numNeuron, boolean isHidden){
        String type;
        Map<Integer, Double[]> temp;
        Double[] d = new Double[3];
        ArrayList<Integer> vals = new ArrayList<>();
        d[0] = biasWeight;
        d[1] = 0.0;
        d[2] = 0.0;

        if(isHidden){
            //numHiddenNeuron = numNeuron;
            type = "hidden";
        }
        else{
            //numOutputNeuron = numNeuron;
            type = "output";
        }
        int counter = neurons.size();
        for (int i = 0; i < numNeuron; i++) {
            if (weights.containsKey(counter+i)){
                temp = weights.get(counter+i);
            }else{
                temp = new HashMap<>();
            }
            temp.put(0, d);
            neurons.put(counter+i, new Neuron(0));
            weights.put(counter+i, temp);
            vals.add(counter+i);
        }
        
        /*int last = neurons.size() - 1;
        last = 5 - 1
        int counter = last + 1;
        counter = 4 + 1
        while(counter <= last + numNeuron) {
        while(5 <= 4+5)
            if (weights.containsKey(counter)) {
                temp = weights.get(counter);
                5
            } else {
                temp = new HashMap<>();
            }

            temp.put(0, d);
            neurons.put(counter, new Neuron(0));
            5
            weights.put(counter, temp);
            5
            vals.add(counter);
            counter++;
            counter 6;
        }*/
        neuronType.put(type, vals);
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
        
        int numInputNeuron = dataSet.numAttributes();
        // input neuron to hidden neuron
        for (int i=1; i<numInputNeuron; i++) {
            for (int j=numInputNeuron; j<(numInputNeuron+numHiddenNeuron); j++) {
                if(weights.containsKey(j)){
                    temp = weights.get(j);
                }else{
                    temp = new HashMap<>();
                }
                temp.put(i, d);
                weights.put(j, temp);
            }
        }

        // hidden to output
        for (int i=numInputNeuron; i<(numInputNeuron+numHiddenNeuron); i++) {
            for (int j =(numInputNeuron+numHiddenNeuron); j<(numInputNeuron+numHiddenNeuron+numOutputNeuron); j++) {
                if (weights.containsKey(j)) {
                    temp = weights.get(j);
                } else {
                    temp = new HashMap<>();
                }
                temp.put(i, d);
                weights.put(j, temp);
            }
        }

        initClassMap();
    }
    
    public void initClassMap() {
        classMap = new HashMap<>();
        ArrayList<Integer> out = neuronType.get("output");

        for (int i=0; i<dataSet.classAttribute().numValues(); i++) {
            classMap.put(out.get(i), (double) i);
        }
    }
    
    
    public ArrayList<Integer> findNodeBefore(int node) {
        return new ArrayList<>(weights.get(node).keySet());
    }

    public ArrayList<Integer> findNodeAfter(int node) {
        ArrayList<Integer> temp = new ArrayList<>();

        weights.entrySet().stream().forEach((weight) -> {
            weight.getValue().entrySet().stream().filter(refs -> refs.getKey() == node).forEach(refs ->
                    temp.add(weight.getKey()));
        });
        return temp;
    }

    public double findWeight(int node1, int node2) {
        return weights.get(node2).get(node1)[0];
    }
    
    public double computeOutputValue(int node, int instanceNo) {
        double temp = 0;
        ArrayList<Integer> refs = findNodeBefore(node);
        ArrayList<Integer> refsAfter = findNodeAfter(node);

        for (Integer ref : refs) {
            temp += findWeight(ref, node) * neurons.get(ref).input.get(instanceNo);
        }

        Neuron n = new Neuron(neurons.get(node));
        n.output = temp;

        for (int i = 0; i < dataSet.numInstances(); i++) {
            n.input.put(i, sigmoid(temp));
        }
        neurons.put(node, n);

        for (Integer ref : refsAfter) {
            n = new Neuron(neurons.get(ref));
            n.input.put(0, sigmoid(temp));
            neurons.put(ref, n);
        }

        return temp;
    }
    
    /**
     * assign instance target value to neuron
     * @param instanceNo instance number
     */
    public void setInstanceTarget(int instanceNo) {
        ArrayList<Integer> outNeuron = neuronType.get("output");
        Neuron n;
        for (int i = 0; i < outNeuron.size(); i++) {
            n = neurons.get(outNeuron.get(i));
            if (Double.compare((double) i, dataSet.instance(instanceNo).classValue()) == 0) {
                n.target.put(instanceNo, 1.0);
            } else {
                n.target.put(instanceNo, 0.0);
            }
            neurons.put(outNeuron.get(i), n);
        }

    }

    /**
     * compute the error of the output layer
     * @param instanceNo the instance related
     */
    public void computeOutputNeuronError(int instanceNo) {
        ArrayList<Integer> outNeuron = neuronType.get("output");
        Neuron n;
        for (Integer anOutNeuron : outNeuron) {
            n = neurons.get(anOutNeuron);
            n.error = n.input.get(0) * (1 - n.input.get(0)) * (n.target.get(instanceNo) - n.input.get(0));
            neurons.put(anOutNeuron, n);
        }
    }
    /**
     * compute the error of the hidden layer
     * @param instanceNo the instance related
     */
    public void computeHiddenNeuronError(int instanceNo) {
        ArrayList<Integer> hiddenNeuron = neuronType.get("hidden");
        ArrayList<Integer> nodeAfter;
        double temp;

        Neuron n;
        for (Integer aHiddenNeuron : hiddenNeuron) {
            n = neurons.get(aHiddenNeuron);
            n.error = n.input.get(instanceNo) * (1 - n.input.get(instanceNo));

            nodeAfter = findNodeAfter(aHiddenNeuron);
            temp = 0;
            for (Integer outNode : nodeAfter) {
                temp += (neurons.get(outNode).error * findWeight(aHiddenNeuron, outNode));
            }
            n.error *= temp;

            neurons.put(aHiddenNeuron, n);
        }
    }
    /**
     * add error (t-o) to the error container
     * @param instanceNo instance related
     * @param diffTargetOutput target minus output
     */
    public void addError(int instanceNo, double diffTargetOutput) {
        error.add(instanceNo, diffTargetOutput);
    }

    /**
     * compute the Mean SquareRoot Error based on the equation: 0.5 * (sum (t-o)^2)
     * @return the Mean SquareRoot Error
     */
    public double computeMSE() {
        double temp = 0;
        for (Double anError : error) {
            temp += Math.pow(anError, 2);
        }

        return temp/2;
    }
    
    /**
     * update the weight of each neuron to another neurons
     * @param instanceNo the related instance
     */
    public void updateWeights(int instanceNo) {
        Double[] tempDouble;
        for (Map.Entry<Integer, Map<Integer, Double[]>> weight: weights.entrySet()) {
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
    /**
     * print the neurons data
     */
    public void printNeuron() {
        System.out.println("Neuron - Input Value (Activation) - Target - Net Value - Error");
        for (Map.Entry<Integer, Neuron> neuron : neurons.entrySet()) {
            System.out.print(neuron.getKey() + " ");
            System.out.print(neuron.getValue().input);
            System.out.print(" "+ neuron.getValue().target + " ");
            System.out.print(" [" + neuron.getValue().output + "] ");
            System.out.println(" [" + neuron.getValue().error + "] ");
        }
    }

    /**
     * print the neuron's weights data
     */
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
    }
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
    
    
    //check sigmoid bener/ngga
    public double sigmoid(double value){
        double exp = 1 + Math.exp(-value);
        if(exp != 0){
            return (1/exp);
        }else{
            return 0;
        }
    }
}
