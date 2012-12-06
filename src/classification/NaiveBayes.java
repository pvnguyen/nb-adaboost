package classification;

import java.io.*;
import java.util.*;
import java.util.Map.*;

/**
 * Naive Bayes classifier
 * 
 * @author phuong
 *
 */
public class NaiveBayes extends Classifier {
	private Map<String, Integer> classCntr = new HashMap<String, Integer>();
	private List<Map<String, Map<String, Integer>>> featureValueClassCntr 
		= new ArrayList<Map<String, Map<String, Integer>>>();

	public static void main(String[] args) {

		File trainingFile = new File(args[0]);
		File testingFile = new File(args[1]);
		NaiveBayes nb = new NaiveBayes();

		try {
			ArrayList<Vector<String>> trainingData = new ArrayList<Vector<String>>();
			ArrayList<Vector<String>> testingData = new ArrayList<Vector<String>>();
			// parse data
			trainingData = nb.parseData(trainingFile);
			testingData = nb.parseData(testingFile);

			// build classifier
			nb.buildClassifier(trainingData);
			// test classifier
			nb.testClassifier(testingData, false);

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

	/**
	 * Build classifier based on training data
	 * 
	 * @param trainingData
	 */
	public void buildClassifier(List<Vector<String>> trainingData) {

		for (Vector<String> vector : trainingData) {
			String classLbl = vector.get(0);

			//Count the number of instances in each class
			if (classCntr.containsKey(classLbl)) {
				classCntr.put(classLbl, classCntr.get(classLbl) + 1);
			} else {
				classCntr.put(classLbl, new Integer(1));
			}
			
			for (int i = 1; i < vector.size(); i++) {
				String featureValuei = vector.get(i);

				//For each feature, store the counter of value-class pairs 
				Map<String, Map<String, Integer>> valueClassCntr;
				//Check if counter of feature i-th already exists
				if (featureValueClassCntr.size() >= i) {
					valueClassCntr = featureValueClassCntr.get(i - 1);
				} else {
					valueClassCntr = new HashMap<String, Map<String, Integer>>();
					featureValueClassCntr.add(valueClassCntr);
				}

				Map<String, Integer> classInValueCntr;
				//Check if this value of feature i-th already exists
				if (valueClassCntr.containsKey(featureValuei)) {
					classInValueCntr = valueClassCntr.get(featureValuei);
				} else {
					classInValueCntr = new HashMap<String, Integer>();
					valueClassCntr.put(featureValuei, classInValueCntr);
				}

				//Store the count of instances for this value
				if (classInValueCntr.containsKey(classLbl)) {
					classInValueCntr.put(classLbl,
							classInValueCntr.get(classLbl) + 1);
				} else {
					classInValueCntr.put(classLbl, 1);
				}
			}
		}
	}

	/* (non-Javadoc)
	 * @see classification.Classifier#classifyInstance(java.util.Vector)
	 */
	public String classifyInstance(Vector<String> vector) {
		double curMaxPrediction = -1;
		String predictedLabel = "";

		//Calculate prediction for each class
		for (Entry<String, Integer> entry : classCntr.entrySet()) {
			String classLbl = entry.getKey();
			int numOfClassInstances = entry.getValue();

			double prediction = 1;
			for (int i = 1; i < vector.size(); i++) {
				Map<String, Map<String, Integer>> valueClassCntr = featureValueClassCntr
						.get(i - 1);
				Map<String, Integer> classInValueCntr = valueClassCntr
						.get(vector.get(i));

				int numValueByClass = 0;
				int totalNumClassInstances = numOfClassInstances;
				if (classInValueCntr != null
						&& classInValueCntr.containsKey(classLbl)) {
					numValueByClass = classInValueCntr.get(classLbl);
				} else {
					numValueByClass = 1;
					totalNumClassInstances++;
				}

				prediction *= ((double) numValueByClass / totalNumClassInstances);
			}

			if (prediction > curMaxPrediction) {
				curMaxPrediction = prediction;
				predictedLabel = classLbl;
			}
		}

		return predictedLabel;
	}
}
