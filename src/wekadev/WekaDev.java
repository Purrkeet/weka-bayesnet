/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekadev;
import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
 
/**
 *
 * @author emedina
 */
public class WekaDev {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
         // Declare two numeric attributes
         Attribute Attribute1 = new Attribute("firstNumeric");
         Attribute Attribute2 = new Attribute("secondNumeric");
          
         // Declare a nominal attribute along with its values
         FastVector fvNominalVal = new FastVector(3);
         fvNominalVal.addElement("blue");
         fvNominalVal.addElement("gray");
         fvNominalVal.addElement("black");
         Attribute Attribute3 = new Attribute("aNominal", fvNominalVal);
          
         // Declare the class attribute along with its values
         FastVector fvClassVal = new FastVector(2);
         fvClassVal.addElement("positive");
         fvClassVal.addElement("negative");
         Attribute ClassAttribute = new Attribute("theClass", fvClassVal);
          
         // Declare the feature vector
         FastVector fvWekaAttributes = new FastVector(4);
         fvWekaAttributes.addElement(Attribute1);    
         fvWekaAttributes.addElement(Attribute2);    
         fvWekaAttributes.addElement(Attribute3);    
         fvWekaAttributes.addElement(ClassAttribute);
          
         // Create an empty training set
         Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);       
          
         // Set class index
         isTrainingSet.setClassIndex(3);
          
         // Create the instance
         Instance iExample = new DenseInstance(4);
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 1.0);      
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 0.5);      
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "positive");
         isTrainingSet.add(iExample);
         iExample = new DenseInstance(4);
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 0.0);      
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 1.5);      
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "blue");
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "negative");
         isTrainingSet.add(iExample);
         iExample = new DenseInstance(4);
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 2.0);      
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 1.5);      
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");
         iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "negative");
         // add the instance
         isTrainingSet.add(iExample);
          // Create a naïve bayes classifier 
          arffReader lector = new arffReader();
          String currentPath = new File("").getAbsolutePath();
          System.out.println(currentPath);
          isTrainingSet = lector.read("/home/emedina/Documents/WekaDev/src/arff/Autor_2002-04.arff");
          
         Classifier cModel = (Classifier)new NaiveBayes();   
        try {
            cModel.buildClassifier(isTrainingSet);
        } catch (Exception ex) {
            Logger.getLogger(WekaDev.class.getName()).log(Level.SEVERE, null, ex);
        }
 
         // Test the model
         Evaluation eTest = null;
        try {
            eTest = new Evaluation(isTrainingSet);
            eTest.evaluateModel(cModel, isTrainingSet);
        } catch (Exception ex) {
            Logger.getLogger(WekaDev.class.getName()).log(Level.SEVERE, null, ex);
        }
         
          
         // Print the result  Weka explorer:
         String strSummary = eTest.toSummaryString();
         System.out.println(strSummary);
          
         // Get the confusion matrix
         double[][] cmMatrix = eTest.confusionMatrix();
         for(int row_i=0; row_i<cmMatrix.length; row_i++){
             for(int col_i=0; col_i<cmMatrix.length; col_i++){
                 System.out.print(cmMatrix[row_i][col_i]);
                 System.out.print("|");
             }
             System.out.println();
         }
         
         //Using the classifier
         //new instances
         // Create the instance
        /*Instance iUse = new DenseInstance(4);
        iUse.setValue((Attribute)fvWekaAttributes.elementAt(0), 1.0);
        iUse.setValue((Attribute)fvWekaAttributes.elementAt(1), 0.5);
        iUse.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");
        //iUse.setValue((Attribute)fvWekaAttributes.elementAt(3), "positive");
                // Specify that the instance belong to the training set
        // in order to inherit from the set description
        iUse.setDataset(isTrainingSet);

        // Get the likelihood of each classes
        // fDistribution[0] is the probability of being “positive”
        // fDistribution[1] is the probability of being “negative”
        double[] fDistribution = null;
        try {
            fDistribution = cModel.distributionForInstance(iUse);
        } catch (Exception ex) {
            Logger.getLogger(WekaDev.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println("the probability of being positive: " + fDistribution[0]);
        System.out.println("the probability of being negative: " + fDistribution[1]);
        */
    }
    
}
