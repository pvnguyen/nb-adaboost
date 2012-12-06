package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

/**
 * Abstract class for classifiers
 * 
 * @author phuong
 * 
 */
public abstract class Classifier {
	public abstract String classifyInstance(Vector<String> vector);

	/**
	 * Parse data from file
	 * 
	 * @param fileName
	 * @return
	 * @throws NumberFormatException
	 * @throws IOException
	 */
	public ArrayList<Vector<String>> parseData(File fileName)
			throws NumberFormatException, IOException {
		FileInputStream fs = new FileInputStream(fileName);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs));
		String strLine;
		ArrayList<Vector<String>> data = new ArrayList<Vector<String>>();
		while ((strLine = br.readLine()) != null) {
			String[] values = strLine.split("\t");

			Vector<String> vector = new Vector<String>();
			for (int i = 0; i < values.length; i++) {
				vector.add(values[i]);
			}
			data.add(vector);
		}
		return data;
	}

	/**
	 * Test classifier, given testing data
	 * 
	 * @param testingData
	 * @param printResults
	 * @throws IOException
	 */
	public void testClassifier(List<Vector<String>> testingData,
			boolean printResults) throws IOException {
		int truePos = 0;
		int trueNeg = 0;
		int falsePos = 0;
		int falseNeg = 0;

		for (Vector<String> vector : testingData) {
			String trueClassLbl = vector.get(0);
			String predictedClassLbl = classifyInstance(vector);

			if (predictedClassLbl.equals("+1") && trueClassLbl.equals("+1"))
				truePos++;
			if (predictedClassLbl.equals("+1") && trueClassLbl.equals("-1"))
				falsePos++;
			if (predictedClassLbl.equals("-1") && trueClassLbl.equals("+1"))
				falseNeg++;
			if (predictedClassLbl.equals("-1") && trueClassLbl.equals("-1"))
				trueNeg++;
		}

		System.out.println(truePos + "\n" + falseNeg + "\n" + falsePos + "\n"
				+ trueNeg);

		if (printResults) {
			System.out.println("Accuracy: "
					+ (((double) truePos + trueNeg) / (trueNeg + truePos
							+ falseNeg + falsePos)));
			System.out.println("Error Rate: "
					+ (((double) falsePos + falseNeg) / (trueNeg + truePos
							+ falseNeg + falsePos)));
			double recall = ((double) truePos / (truePos + falseNeg));
			System.out.println("Sensitivity: " + recall);
			System.out.println("Specificity: "
					+ ((double)trueNeg / (trueNeg + falsePos)));
			double precision = ((double)truePos / (truePos + falsePos));
			System.out.println("Precision: " + precision);
			System.out.println("F1: " + 2 * precision * recall
					/ (precision + recall));
			System.out.println("Fbeta (0.5): " + (1 + 0.25) * precision
					* recall / (0.25*precision + recall));
			System.out.println("Fbeta (2): " + (1 + 4) * precision
					* recall / (4*precision + recall));
		}
	}
}
