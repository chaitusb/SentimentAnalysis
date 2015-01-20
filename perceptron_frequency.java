package org.sbu.nlp.homework1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class perceptron_frequency {
	static int globalCounter = 0;
	static Map<String, Integer> wordlist = new HashMap<String, Integer>();
	static int wordlistCount;
	static int[][] posMatrix = null;
	static int[][] negMatrx = null;

	public static void main(String[] args) throws IOException {
		final File posFolder = new File("pos");
		final File negFolder = new File("neg");
		countDistinctWords(posFolder);
		countDistinctWords(negFolder);
		posMatrix = new int[1000][wordlist.size()];
		negMatrx = new int[1000][wordlist.size()];
		listFilesForFolder(posFolder, "pos");
		listFilesForFolder(negFolder, "neg");
		wordlistCount = wordlist.size();
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
			weightVectorCalculation(posList, negList, posTestMatrix,
					negTestMatrix);
		}
	}

	public static void weightVectorCalculation(
			List<double[][]> postrainingMatrix,
			List<double[][]> negTrainingMatrix, double[][] testPosMatrix,
			double[][] testNegMatrix) {

		double[][] weightVector = new double[1][wordlist.size()];
		Random randomGenerator = new Random();
		int loopCounter = 0;
		for (int i = 0; i < wordlist.size(); i++) {
			weightVector[0][i] = 0.0;
		}
		int mistakes = 1;
		while (mistakes != 0) {
			loopCounter++;
			mistakes = 0;
			for (int k = 0; k < 4; k++) {
				for (int i = 0; i < 200; i++) {
					double weightedSum = 0.0;
					for (int j = 0; j < wordlist.size(); j++) {
						weightedSum += postrainingMatrix.get(k)[i][j]
								* weightVector[0][j];
					}
					if (weightedSum < 0) {
						mistakes++;
						for (int j = 0; j < wordlist.size(); j++) {
							weightVector[0][j] = weightVector[0][j]
									+ postrainingMatrix.get(k)[i][j];
						}
					}
				}
				for (int i = 0; i < 200; i++) {
					double weightedSum = 0.0;
					for (int j = 0; j < wordlist.size(); j++) {
						weightedSum += negTrainingMatrix.get(k)[i][j]
								* weightVector[0][j];
					}
					if (weightedSum >= 0) {
						mistakes++;
						for (int j = 0; j < wordlist.size(); j++) {
							weightVector[0][j] = weightVector[0][j]
									- negTrainingMatrix.get(k)[i][j];
						}
					}
				}
			}
		}
		int misclassification = 0;
		for (int i = 0; i < 200; i++) {
			double weightedsum = 0.0;
			for (int j = 0; j < wordlist.size(); j++) {
				weightedsum += weightVector[0][j] * testPosMatrix[i][j];
			}
			if (weightedsum < 0) {
				misclassification++;
			}
		}
		for (int i = 0; i < 200; i++) {
			double weightedsum = 0.0;
			for (int j = 0; j < wordlist.size(); j++) {
				weightedsum += weightVector[0][j] * testNegMatrix[i][j];
			}
			if (weightedsum >= 0) {
				misclassification++;
			}
		}
		System.out.println("Prediction Accuracy = "
				+ (100 - (double) misclassification / 4));
		System.out.println("Error Rate = " + (double) misclassification / 4);
	}

	public static void countDistinctWords(final File folder)
			throws FileNotFoundException, UnsupportedEncodingException {
		String currentLine = "";
		for (final File fileEntry : folder.listFiles()) {
			BufferedReader bReader = new BufferedReader(new FileReader(
					fileEntry));
			try {
				while ((currentLine = bReader.readLine()) != null) {
					currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
					currentLine = currentLine.replaceAll("\\s+", " ");
					currentLine = currentLine.toLowerCase();
					String[] temp = currentLine.split(" ");
					for (int index = 0; index < temp.length; index++) {
						if (temp[index].trim().length() > 0) {
							if (!wordlist.containsKey(temp[index])) {
								wordlist.put(temp[index], globalCounter);
								globalCounter++;
							}
						}
					}
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public static void listFilesForFolder(final File folder, String type)
			throws FileNotFoundException, UnsupportedEncodingException {
		String currentLine = "";
		int counter = -1;
		for (final File fileEntry : folder.listFiles()) {
			Map<String, Integer> wordCount = new HashMap<String, Integer>();
			counter++;
			try {
				BufferedReader bReader = new BufferedReader(new FileReader(
						fileEntry));
				while ((currentLine = bReader.readLine()) != null) {
					currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
					currentLine = currentLine.replaceAll("\\s+", " ");
					currentLine = currentLine.toLowerCase();
					String[] temp = currentLine.split(" ");
					for (int index = 0; index < temp.length; index++) {
						if (temp[index].trim().length() > 1) {
							if (wordCount.containsKey(temp[index])) {
								wordCount.put(temp[index],
										wordCount.get(temp[index]) + 1);
							} else {
								wordCount.put(temp[index], 1);
							}
						}
					}
				}
				Iterator it = wordCount.entrySet().iterator();
				while (it.hasNext()) {
					Map.Entry pairs = (Map.Entry) it.next();
					if (type.equals("pos")) {
						posMatrix[counter][wordlist.get(pairs.getKey())] = (int) pairs
								.getValue();
					} else {
						negMatrx[counter][wordlist.get(pairs.getKey())] = (int) pairs
								.getValue();
					}
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
