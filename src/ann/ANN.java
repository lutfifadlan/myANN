/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
/**
 *
 * @author Mochamad Lutfi F
 */
public class ANN {
    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        // TODO code application logic here
        DataSource source = new DataSource("D:/ITB/Semester 7/Machine Learning/Tugas Besar ANN/dataset.arff");
        Instances data = source.getDataSet();
        // setting class attribute
        //data.setClassIndex(classIndex);
        int currentEpoch = 0;
        int ninput = 4;
        int ninstance = 1;
        int target = -1;
        Perceptron Perceptron = new Perceptron(ninput,ninstance);
        double[] listWeight = new double[ninput];
        double[] instance = new double[ninput];
        double[] listTarget = new double[ninstance];
        instance[0] = 1;
        instance[1] = 1;
        instance[2] = 0;
        instance[3] = 1;
        listTarget[0] = -1;
   //     listTarget[1] = 1;
     //   listTarget[2] = 1;
        for (int i=0; i<ninput;i++){
            listWeight[i] = 0;
        }
        Perceptron.setInstance(instance);
        Perceptron.setListWeight(listWeight);
        Perceptron.setListTarget(listTarget);
        Perceptron.PerceptronLearning(instance,listWeight);
        Perceptron.printPerEpoch();
        Perceptron.EpochStatus(currentEpoch);
        if(!Perceptron.getIsConvergen())
        {
            for(int i=0; i<ninput; i++)
                System.out.println("allWeightUpdated[" + i + "]" + " = " + Perceptron.getAllWeightUpdated().get(i));
        }
            //Perceptron.PerceptronLearning(instance,listWeight);
    }
}
