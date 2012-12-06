package classification;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;

/**
 * Naive Bayes using Adaboost
 * 
 * @author phuong
 * 
 */
public class NBAdaboost extends Classifier {
	private static Map<Integer, Double> classifierErrors = new HashMap<Integer, Double>();
	private static List<NaiveBayes> classifiers = new ArrayList<NaiveBayes>();
	private static final int NUM_OF_ITERATIONS = 20;
	private static final int NUM_OF_ERROR_TRIALS = 10;

	public static void main(String[] args) {
		File trainingFile = new File(args[0]);
		File testingFile = new File(args[1]);

		try {
			NBAdaboost nba = new NBAdaboost();

			// parse data from files
			List<Vector<String>> trainingData = nba.parseData(trainingFile);
			List<Vector<String>> testingData = nba.parseData(testingFile);

			// build classifier
			nba.buildClassifier(trainingData, NUM_OF_ITERATIONS);
			// test classifier
			nba.testClassifier(testingData, false);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

	/**
	 * Weighted random sampling data for training
	 * 
	 * @param trainingData
	 * @param weights
	 * @return
	 */
	private List<Vector<String>> randomSampling(
			List<Vector<String>> trainingData, double[] weights) {
		List<Vector<String>> sampledData = new ArrayList<Vector<String>>();
		NavigableMap<Double, Integer> weightRanges = new TreeMap<Double, Integer>();
		Double totalWeight = 0.0;

		for (int i = 0; i < weights.length; i++) {
			totalWeight += weights[i];
			weightRanges.put(totalWeight, i);
		}
		Random random = new Random();
		for (int i = 0; i < trainingData.size(); i++) {
			double nextSampleID = random.nextDouble() * totalWeight;
			sampledData.add(trainingData.get(weightRanges.ceilingEntry(
					nextSampleID).getValue()));
		}

		return sampledData;
	}

	/**
	 * Build ensemble classifier, given training data
	 * 
	 * @param trainingData
	 * @param numOfIterations
	 */
	private void buildClassifier(List<Vector<String>> trainingData,
			int numOfIterations) {

		double[] weights = new double[trainingData.size()];
		for (int i = 0; i < trainingData.size(); i++)
			weights[i] = (double) 1 / trainingData.size();

		for (int i = 1; i <= numOfIterations; i++) {
			double errorRate = 0;
			boolean[] isCorrect = new boolean[trainingData.size()];
			NaiveBayes nb;
			List<Vector<String>> sampledData;
			int numOfTrials = 0;
			do {
				sampledData = randomSampling(trainingData, weights);

				nb = new NaiveBayes();
				nb.buildClassifier(sampledData);

				for (int j = 0; j < trainingData.size(); j++) {
					Vector<String> vector = trainingData.get(j);
					String predicted = nb.classifyInstance(vector);

					if (predicted.equals(vector.get(0))) {
						isCorrect[j] = true;
					} else {
						errorRate += weights[j];
					}
				}
				numOfTrials++;
			} while (errorRate > 0.5 && numOfTrials != NUM_OF_ERROR_TRIALS);

			// Stop boosting when the errorRate is too high or 0
			if (errorRate == 0 || errorRate > 0.5) {
				if (classifiers.size() == 0) {
					classifiers.add(nb);
					classifierErrors.put(0, errorRate);
				}
				break;
			}

			double sumWeights = 0.0;
			for (int j = 0; j < sampledData.size(); j++) {
				if (isCorrect[j]) {
					weights[j] = weights[j] * errorRate / (1 - errorRate);
				}
				sumWeights += weights[j];
			}

			for (int j = 0; j < sampledData.size(); j++) {
				weights[j] = weights[j] / sumWeights;
			}

			classifiers.add(nb);
			classifierErrors.put(classifiers.size() - 1, errorRate);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see classification.Classifier#classifyInstance(java.util.Vector)
	 */
	public String classifyInstance(Vector<String> vector) {
		Map<String, Double> labelVotes = new HashMap<String, Double>();
		int classifierID = 0;

		for (NaiveBayes classifier : classifiers) {
			String label = classifier.classifyInstance(vector);
			double errorRate = classifierErrors.get(classifierID);
			double voteConfidence = Math.log((1 - errorRate) / errorRate);

			if (labelVotes.containsKey(label)) {
				labelVotes.put(label, labelVotes.get(label) + voteConfidence);
			} else {
				labelVotes.put(label, new Double(voteConfidence));
			}
			classifierID++;
		}

		double max = -1.0;
		String predictedLabel = "";
		for (String label : labelVotes.keySet()) {
			double votes = labelVotes.get(label);
			if (votes > max) {
				max = votes;
				predictedLabel = label;
			}
		}

		return predictedLabel;
	}
}
