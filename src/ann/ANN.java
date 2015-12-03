/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package ann;
import java.io.IOException;
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
