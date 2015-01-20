package org.sbu.nlp.homework1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.L2R_L2_SvrFunction;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

public class Liblinear {

	static double[] GROUPS_ARRAY = new double[1600];

	static Map<String, Integer> uniqueWordList = new TreeMap<String, Integer>();
	static int globalCounter = 0;

	static TreeMap<Integer, TreeMap<Integer, Double>> featuresData = new TreeMap<Integer, TreeMap<Integer, Double>>();
	static TreeMap<Integer, TreeMap<Integer, Double>> featuresTraining = new TreeMap<Integer, TreeMap<Integer, Double>>();
	static TreeMap<Integer, TreeMap<Integer, Double>> featuresTesting = new TreeMap<Integer, TreeMap<Integer, Double>>();
	static Feature[][] featureTrainingVector = new Feature[1600][];

	public static void main(String[] args) throws IOException {
		final File posFolder = new File("pos");
		final File negFolder = new File("neg");
		getUniqueWordCount(posFolder);
		getUniqueWordCount(negFolder);
		getfeaturesmap(posFolder, negFolder);
		for (int i = 0; i < 5; i++) {
			generateFold(i);

			for (int label = 0; label < 1600; label++) {
				if (label < 800) {
					GROUPS_ARRAY[label] = 1.0;
				} else {
					GROUPS_ARRAY[label] = 0.0;
				}
			}

			for (int it = 0; it < 1600; it++) {
				TreeMap<Integer, Double> temp = featuresTraining.get(it);
				Feature[] featureArray = new FeatureNode[temp.size()];
				Iterator itr = temp.entrySet().iterator();
				int j = 0;
				while (itr.hasNext()) {
					Map.Entry pairs = (Map.Entry) itr.next();

					FeatureNode featureNode = new FeatureNode(
							(int) pairs.getKey() + 1, (double) pairs.getValue());
					featureArray[j] = featureNode;
					j++;
				}
				featureTrainingVector[it] = featureArray;
			}
			predict();
		}
	}

	private static void generateFold(int i) {
		if (i == 0) {
			for (int j = 0; j < 2000; j++) {
				if (j < 200)
					featuresTesting.put(j, featuresData.get(j));
				if (j >= 200 && j < 1000)
					featuresTraining.put(j - 200, featuresData.get(j));
				if (j >= 1000 && j < 1200)
					featuresTesting.put(j - 800, featuresData.get(j));
				if (j >= 1200)
					featuresTraining.put(j - 400, featuresData.get(j));
			}
		}
		if (i == 1) {
			for (int j = 0; j < 2000; j++) {
				if (j < 200)
					featuresTraining.put(j, featuresData.get(j));
				if (j >= 200 && j < 400)
					featuresTesting.put(j - 200, featuresData.get(j));
				if (j >= 400 && j < 1200)
					featuresTraining.put(j - 200, featuresData.get(j));
				if (j >= 1200 && j < 1400)
					featuresTesting.put(j - 1000, featuresData.get(j));
				if (j >= 1200)
					featuresTraining.put(j - 200, featuresData.get(j));
			}
		}
		if (i == 2) {
			for (int j = 0; j < 2000; j++) {
				if (j < 400)
					featuresTraining.put(j, featuresData.get(j));
				if (j >= 400 && j < 600)
					featuresTesting.put(j - 400, featuresData.get(j));
				if (j >= 600 && j < 1400)
					featuresTraining.put(j - 200, featuresData.get(j));
				if (j >= 1400 && j < 1600)
					featuresTesting.put(j - 1200, featuresData.get(j));
				if (j >= 1600)
					featuresTraining.put(j - 400, featuresData.get(j));
			}
		}
		if (i == 3) {
			for (int j = 0; j < 2000; j++) {
				if (j < 600)
					featuresTraining.put(j, featuresData.get(j));
				if (j >= 600 && j < 800)
					featuresTesting.put(j - 600, featuresData.get(j));
				if (j >= 800 && j < 1600)
					featuresTraining.put(j - 200, featuresData.get(j));
				if (j >= 1600 && j < 1800)
					featuresTesting.put(j - 1400, featuresData.get(j));
				if (j >= 1800)
					featuresTraining.put(j - 400, featuresData.get(j));
			}
		}
		if (i == 4) {
			for (int j = 0; j < 2000; j++) {
				if (j < 800)
					featuresTraining.put(j, featuresData.get(j));
				if (j >= 800 && j < 1000)
					featuresTesting.put(j - 800, featuresData.get(j));
				if (j >= 1000 && j < 1800)
					featuresTraining.put(j - 200, featuresData.get(j));
				if (j >= 1800)
					featuresTesting.put(j - 1600, featuresData.get(j));
			}
		}
	}

	private static void predict() throws IOException {
		Problem prob = new Problem();
		prob.l = 1600;
		prob.n = uniqueWordList.size();
		prob.x = featureTrainingVector;
		prob.y = GROUPS_ARRAY;
		SolverType solver = SolverType.L2R_LR;
		double C = 1.0;
		double eps = 0.01;
		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(prob, parameter);
		File modelFile = new File("model");
		model.save(modelFile);
		model = Model.load(modelFile);
		int miscalculation = 0;
		for (int i = 0; i < 400; i++) {
			TreeMap<Integer, Double> temp = featuresTesting.get(i);
			Feature[] featureArray = new FeatureNode[temp.size()];
			Iterator itr = temp.entrySet().iterator();
			int j = 0;
			while (itr.hasNext()) {
				Map.Entry pairs = (Map.Entry) itr.next();
				FeatureNode featureNode = new FeatureNode(
						(int) pairs.getKey() + 1, (double) pairs.getValue());
				featureArray[j] = featureNode;
				j++;
			}
			double prediction = Linear.predict(model, featureArray);
			if (i < 200 && prediction == 0.0) {
				miscalculation++;
			}
			if (i > 200 && prediction == 1.0) {
				miscalculation++;
			}
		}
		System.out.println("Prediction Accuracy = " + (100 - (double) miscalculation / 4));
		System.out.println("Error Rate = " + (double) miscalculation / 4);
	}

	private static void getUniqueWordCount(File folder) throws IOException {
		String currentLine = "";
		int counter = -1;
		for (final File fileEntry : folder.listFiles()) {
			counter++;
			BufferedReader bReader = new BufferedReader(new FileReader(
					fileEntry));
			while ((currentLine = bReader.readLine()) != null) {
				currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
				currentLine = currentLine.replaceAll("\\s+", " ");
				currentLine = currentLine.toLowerCase();
				String[] temp = currentLine.split(" ");
				for (int index = 0; index < temp.length; index++) {
					if (temp[index].trim().length() > 0) {
						if (!uniqueWordList.containsKey(temp[index])) {
							uniqueWordList.put(temp[index], globalCounter);
							globalCounter++;
						}
					}
				}
			}
		}
	}

	private static void getfeaturesmap(File posFolder, File negFolder)
			throws IOException {
		String currentLine = "";
		int counter = 0;
		TreeMap<Integer, Double> wordCount = new TreeMap<Integer, Double>();
		for (final File fileEntry : posFolder.listFiles()) {
			BufferedReader bReader = new BufferedReader(new FileReader(
					fileEntry));
			while ((currentLine = bReader.readLine()) != null) {
				currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
				currentLine = currentLine.replaceAll("\\s+", " ");
				currentLine = currentLine.toLowerCase();
				String[] temp = currentLine.split(" ");
				for (int index = 0; index < temp.length; index++) {
					if (temp[index].trim().length() > 0) {
						if (wordCount.containsKey(uniqueWordList
								.get(temp[index]))) {
							double d = wordCount.get(uniqueWordList
									.get(temp[index]));
							wordCount.put(uniqueWordList.get(temp[index]),
									d + 1.0);
						} else {
							wordCount.put(uniqueWordList.get(temp[index]), 1.0);
						}
					}
				}
			}
			featuresData.put(counter, wordCount);
			counter++;
			wordCount = new TreeMap<>();
		}
		for (final File fileEntry : negFolder.listFiles()) {
			BufferedReader bReader = new BufferedReader(new FileReader(
					fileEntry));
			while ((currentLine = bReader.readLine()) != null) {
				currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
				currentLine = currentLine.replaceAll("\\s+", " ");
				currentLine = currentLine.toLowerCase();
				String[] temp = currentLine.split(" ");
				for (int index = 0; index < temp.length; index++) {
					if (temp[index].trim().length() > 0) {
						if (wordCount.containsKey(uniqueWordList
								.get(temp[index]))) {
							double d = wordCount.get(uniqueWordList
									.get(temp[index]));
							wordCount.put(uniqueWordList.get(temp[index]),
									d + 1.0);
						} else {
							wordCount.put(uniqueWordList.get(temp[index]), 1.0);
						}
					}
				}
			}
			featuresData.put(counter, wordCount);
			counter++;
			wordCount = new TreeMap<>();
		}
	}
}
