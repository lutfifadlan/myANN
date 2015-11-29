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
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
/**
 *
 * @author Mochamad Lutfi F
 */
public class ANN {
    //CSV file header
  /*  private static final String [] FILE_HEADER_MAPPING = {"x0","x1","x2","x3","Target"};
    //dataset attribute
    private static final String x0 = "x0";
    private static final String x1 = "x1";
    private static final String x2 = "x2";
    private static final String x3 = "x3"; 
    private static final String Target = "Target"; */
    
    public static void main(String[] args) throws IOException {
        Perceptron unit = new Perceptron();
        //unit.readARFF("C:\\Program Files\\Weka-3-6\\data\\weather.numeric.arff");
        //unit.readARFF("C:\\Program Files\\Weka-3-6\\data\\weather.numeric.arff");
        System.out.println("oy");
//        unit.PerceptronLearning(unit.getData());
        unit.buildClassifier(unit.getData());
    }
        // TODO code application logic here
      /*  Perceptron unit = null;
        int currentEpoch = 0;
        FileReader fileReader = null;
        CSVParser csvFileParser = null;
        //Create the CSVFormat object with the header mapping
        CSVFormat csvFileFormat = CSVFormat.DEFAULT.withHeader(FILE_HEADER_MAPPING);
        //Create a new list of student to be filled by CSV file data 
        List units = new ArrayList();
        //initialize FileReader object
        fileReader = new FileReader("D:/ITB/Semester 7/Machine Learning/dataset1.csv");
        //initialize CSVParser object
        csvFileParser = new CSVParser(fileReader, csvFileFormat);
        //Get a list of CSV file records
        List csvRecords = csvFileParser.getRecords();
        //Read the CSV file records starting from the second record to skip the header
        List<Double> instance = new ArrayList();
        List<Double> target = new ArrayList();
        for (int i = 1; i < csvRecords.size(); i++) {
            CSVRecord record = (CSVRecord) csvRecords.get(i);
            //Create a new object and fill his data
           // unit = new Perceptron(Double.parseDouble(record.get(x0)), Double.parseDouble(record.get(x1)), Double.parseDouble(record.get(x2)), Double.parseDouble(record.get(x3)), Double.parseDouble(record.get(Target)));
            instance.add(Double.NaN)
            units.add(unit);  
        }*/
        /*
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
        listTarget[1] = 1;
        listTarget[2] = 1;
        for (int i=0; i<ninput;i++){
            listWeight[i] = 0;
        }
        Perceptron.setInstance(instance);
        Perceptron.setListWeight(listWeight);
        Perceptron.setListTarget(listTarget);*/
        /*Perceptron unitNode;
        unitNode = new Perceptron(4,3);
        unitNode.setAllInstance(units.get);
        unitNode.printPerEpoch();
        unitNode.EpochStatus(currentEpoch);
        if(!unitNode.getIsConvergen())
        {
            for(int i=0; i<unit.getNinput(); i++)
                System.out.println("allWeightUpdated[" + i + "]" + " = " + unit.getAllWeightUpdated().get(i));
        }
    }*/
}
