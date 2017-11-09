/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekadev;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 *
 * @author emedina
 */
public class arffReader {
   
    public Instances read(String path){
        FileReader in = null;
        try {
            in = new FileReader(path);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(arffReader.class.getName()).log(Level.SEVERE, null, ex);
        }
        BufferedReader reader = new BufferedReader(in);
        ArffLoader.ArffReader arff = null;
        try {
           arff = new ArffLoader.ArffReader(reader);
        } catch (IOException ex) {
            Logger.getLogger(arffReader.class.getName()).log(Level.SEVERE, null, ex);
        }
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }
}
