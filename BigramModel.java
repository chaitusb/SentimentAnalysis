package org.sbu.nlp.homework1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class BigramModel {

	static int globalCounter = 0;
	static List<String> wordList = new ArrayList<String>();
	static List<String> bigramList = new ArrayList<String>();
	static Map<String, Integer> uniqueWordList = new HashMap<String, Integer>();
	static Map<String, Integer> wordcountmap = new HashMap<String, Integer>();;
	static List<Map<String, Integer>> poswordcountlist = new ArrayList<Map<String, Integer>>();
	static List<Map<String, Integer>> negwordcountlist = new ArrayList<Map<String, Integer>>();
	static int[][] posMatrix = null;
	static int[][] negMatrx = null;
	static int wordlistCount;

	public static void main(String[] args) throws IOException {
		final File posFolder = new File("pos");
		final File negFolder = new File("neg");
		getUniqueWordCount(posFolder);
		getUniqueWordCount(negFolder);
		posMatrix = new int[1000][uniqueWordList.size()];
		negMatrx = new int[1000][uniqueWordList.size()];
		getBigrams(posFolder, "pos");
		getBigrams(negFolder, "neg");
		wordlistCount = uniqueWordList.size();
		naiveBayesMatrixCreation();
	}

	public static void naiveBayesMatrixCreation() {
		double[][] foldsMatrixPos0 = new double[200][wordlistCount];
		double[][] foldsMatrixPos1 = new double[200][wordlistCount];
		double[][] foldsMatrixPos2 = new double[200][wordlistCount];
		double[][] foldsMatrixPos3 = new double[200][wordlistCount];
		double[][] foldsMatrixPos4 = new double[200][wordlistCount];
		double[][] foldsMatrixNeg0 = new double[200][wordlistCount];
		double[][] foldsMatrixNeg1 = new double[200][wordlistCount];
		double[][] foldsMatrixNeg2 = new double[200][wordlistCount];
		double[][] foldsMatrixNeg3 = new double[200][wordlistCount];
		double[][] foldsMatrixNeg4 = new double[200][wordlistCount];
		List<double[][]> posMatriceList = new ArrayList<double[][]>();
		List<double[][]> negMatriceList = new ArrayList<double[][]>();
		posMatriceList.add(foldsMatrixPos0);
		negMatriceList.add(foldsMatrixNeg0);
		posMatriceList.add(foldsMatrixPos1);
		negMatriceList.add(foldsMatrixNeg1);
		posMatriceList.add(foldsMatrixPos2);
		negMatriceList.add(foldsMatrixNeg2);
		posMatriceList.add(foldsMatrixPos3);
		negMatriceList.add(foldsMatrixNeg3);
		posMatriceList.add(foldsMatrixPos4);
		negMatriceList.add(foldsMatrixNeg4);
		int fold = 0;
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 200; j++) {
				for (int k = 0; k < wordlistCount; k++) {
					posMatriceList.get(i)[j][k] = posMatrix[i * 200 + j][k];
					negMatriceList.get(i)[j][k] = negMatrx[i * 200 + j][k];
				}
			}
		}
		posMatrix = null;
		negMatrx = null;
		double[][] posTestMatrix = null;
		double[][] negTestMatrix = null;
		for (int i = 0; i < 5; i++) {
			posTestMatrix = posMatriceList.get(i);
			negTestMatrix = negMatriceList.get(i);
			List<double[][]> posList = new ArrayList<double[][]>();
			List<double[][]> negList = new ArrayList<double[][]>();
			for (int j = 0; j < 5; j++) {
				if (i != j) {
					posList.add(posMatriceList.get(j));
					negList.add(negMatriceList.get(j));
				}
			}
			calculateProbability(posList, negList, posTestMatrix,
					negTestMatrix, fold);
			fold++;
		}
	}

	public static void calculateProbability(List<double[][]> postrainingMatrix,
			List<double[][]> negTrainingMatrix, double[][] testPosMatrix,
			double[][] testNegMatrix, int fold) {
		double posTrainingProb[][] = new double[1][wordlistCount];
		double negTrainingProb[][] = new double[1][wordlistCount];
		Map<String, Integer> positiveClassHash = new HashMap<String, Integer>();
		Map<String, Integer> negativeClassHash = new HashMap<String, Integer>();
		for (int j = 0; j < 5; j++) {
			if (fold != j) {
				for (int i = 0; i < 200; i++) {
					Iterator it = poswordcountlist.get(j * 200 + i).entrySet()
							.iterator();
					while (it.hasNext()) {
						Map.Entry pairs = (Map.Entry) it.next();
						if (positiveClassHash.containsKey(pairs.getKey())) {
							positiveClassHash.put((String) pairs.getKey(),
									positiveClassHash.get(pairs.getKey())
											+ (Integer) pairs.getValue());
						} else {
							positiveClassHash.put((String) pairs.getKey(),
									(Integer) pairs.getValue());
						}
					}
					Iterator it1 = negwordcountlist.get(j * 200 + i).entrySet()
							.iterator();
					while (it1.hasNext()) {
						Map.Entry pairs = (Map.Entry) it1.next();
						if (negativeClassHash.containsKey(pairs.getKey())) {
							negativeClassHash.put((String) pairs.getKey(),
									negativeClassHash.get(pairs.getKey())
											+ (Integer) pairs.getValue());
						} else {
							negativeClassHash.put((String) pairs.getKey(),
									(Integer) pairs.getValue());
						}
					}
				}
			}
		}
		int totalPos = 0;
		int totalNeg = 0;
		for (int j = 0; j < wordlistCount; j++) {
			for (int k = 0; k < 200; k++) {
				posTrainingProb[0][j] += postrainingMatrix.get(0)[k][j]
						+ postrainingMatrix.get(1)[k][j]
						+ postrainingMatrix.get(2)[k][j]
						+ postrainingMatrix.get(3)[k][j];
				negTrainingProb[0][j] += negTrainingMatrix.get(0)[k][j]
						+ negTrainingMatrix.get(1)[k][j]
						+ negTrainingMatrix.get(2)[k][j]
						+ negTrainingMatrix.get(3)[k][j];
			}
		}
		for (int i = 0; i < wordlistCount; i++) {
			totalPos += posTrainingProb[0][i];
			totalNeg += negTrainingProb[0][i];
		}
		for (int i = 0; i < wordlistCount; i++) {
			posTrainingProb[0][i] = (posTrainingProb[0][i] + 1)
					/ (totalPos + wordlistCount);
			negTrainingProb[0][i] = (negTrainingProb[0][i] + 1)
					/ (totalNeg + wordlistCount);
		}
		int misClassification = 0;
		for (int i = 0; i < 200; i++) {
			Iterator it = poswordcountlist.get(fold * 200 + i).entrySet()
					.iterator();
			double sum1 = 0.0;
			double sum2 = 0.0;
			double a = 0;
			double b = 0;
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry) it.next();
				String[] splits = pairs.getKey().toString().split("\\|");
				if (positiveClassHash.containsKey(pairs.getKey())) {
					a = positiveClassHash.get(pairs.getKey());
					b = posTrainingProb[0][Integer.parseInt(splits[0])];
					sum1 += Math.log((a + 1) / (b + uniqueWordList.size()));
				} else {
					a = 1;
					b = posTrainingProb[0][Integer.parseInt(splits[0])];
					sum1 += Math.log(a / b + uniqueWordList.size());
				}
				if (negativeClassHash.containsKey(pairs.getKey())) {
					a = negativeClassHash.get(pairs.getKey());
					b = negTrainingProb[0][Integer.parseInt(splits[0])];
					sum2 += Math.log((a + 1) / (b + uniqueWordList.size()));
				} else {
					a = 1;
					b = (negTrainingProb[0][Integer.parseInt(splits[0])] + wordlistCount);
					sum2 += Math.log(a / (b + uniqueWordList.size()));
				}
			}
			if (sum1 > sum2) {
				misClassification++;
			}
		}
		for (int i = 0; i < 200; i++) {
			Iterator it = negwordcountlist.get(fold * 200 + i).entrySet()
					.iterator();
			double sum1 = 0.0;
			double sum2 = 0.0;
			double a = 0;
			double b = 0;
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry) it.next();
				String[] splits = pairs.getKey().toString().split("\\|");
				if (positiveClassHash.containsKey(pairs.getKey())) {
					a = positiveClassHash.get(pairs.getKey());
					b = posTrainingProb[0][Integer.parseInt(splits[0])];
					sum1 += Math.log((double) (a + 1) / (b + uniqueWordList.size()));
				} else {
					a = 1;
					b = posTrainingProb[0][Integer.parseInt(splits[0])]
							+ wordlistCount;
					sum1 += Math.log((double) 1 / (b + uniqueWordList.size()));
				}
				if (negativeClassHash.containsKey(pairs.getKey())) {
					a = negativeClassHash.get(pairs.getKey());
					b = negTrainingProb[0][Integer.parseInt(splits[0])];
					sum2 += Math.log((double)(a + 1) / (b + uniqueWordList.size()));
				} else {
					a = 1;
					b = negTrainingProb[0][Integer.parseInt(splits[0])]
							+ wordlistCount;
					sum2 += Math.log((double) a / (b + uniqueWordList.size()));
				}
			}
			if (sum1 < sum2) {
				misClassification++;
			}
		}
		System.out.println("Prediction Accuracy = " + (100 - (double) misClassification / 4));
		System.out.println("Error Rate = " + (double) misClassification / 4);
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
				String[] temp = currentLine.toString().trim().split(" ");
				for (int index = 0; index < temp.length; index++) {
					if (temp[index].trim().length() > 0) {
						wordList.add(temp[index]);
						if (!uniqueWordList.containsKey(temp[index])) {
							uniqueWordList.put(temp[index], globalCounter);
							globalCounter++;
						}
					}
				}
			}
		}
	}

	private static void getBigrams(File folder, String type) throws IOException {
		String currentLine = "";
		int counter = -1;
		for (final File fileEntry : folder.listFiles()) {
			counter++;
			wordList = new ArrayList<String>();
			BufferedReader bReader = new BufferedReader(new FileReader(
					fileEntry));
			while ((currentLine = bReader.readLine()) != null) {
				currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
				currentLine = currentLine.replaceAll("\\s+", " ");
				currentLine = currentLine.toLowerCase();
				String[] temp = currentLine.toString().trim().split(" ");
				for (int index = 0; index < temp.length; index++) {
					if (temp[index].trim().length() > 0) {
						wordList.add(temp[index].trim());
						if (!uniqueWordList.containsKey(temp[index])) {
							uniqueWordList.put(temp[index], globalCounter);
							globalCounter++;
						}
					}
				}
			}
			for (int j = 1; j < wordList.size(); j++) {
				String bigram = uniqueWordList.get(wordList.get(j - 1)) + "|"
						+ uniqueWordList.get(wordList.get(j));
				if (!wordcountmap.containsKey(bigram)) {
					wordcountmap.put(bigram, 1);
				} else {
					wordcountmap.put(bigram, wordcountmap.get(bigram) + 1);
				}
			}
			if (type == "pos") {
				poswordcountlist.add(wordcountmap);
			} else {
				negwordcountlist.add(wordcountmap);
			}
			wordcountmap = new HashMap<String, Integer>();
			Iterator it = uniqueWordList.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry) it.next();
				if (type.equals("pos")) {
					posMatrix[counter][uniqueWordList.get(pairs.getKey())] = (int) pairs
							.getValue();
				} else {
					negMatrx[counter][uniqueWordList.get(pairs.getKey())] = (int) pairs
							.getValue();
				}
			}
		}
	}
}