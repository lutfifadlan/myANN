/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import weka.classifiers.Evaluation;
/**
 *
 * @author Mochamad Lutfi F
 */
public class ANN { 
    public static void main(String[] args) throws IOException, Exception {
        Perceptron unit = new Perceptron();
        //Incremental unit = new Incremental();
        unit.buildClassifier(unit.getData());
        Evaluation eval; 
        eval = new Evaluation(unit.getData());
        eval.evaluateModel(unit, unit.getData());
        System.out.println(eval.toSummaryString());
        //for(int i=0; i<unit.getNInstance();i++)
          //  unit.classifyInstance(unit.getData().instance(i));
        
      /*  int ninstance = unit.getData().numInstances();
        for(int i=0; i<ninstance; i++)
            unit.classifyInstance(unit.getData().instance(i)); */
    }
}
