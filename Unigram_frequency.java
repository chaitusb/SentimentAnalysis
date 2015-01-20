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

public class Unigram_frequency {

	static int globalCounter = 0;
	static Map<String, Integer> wordlist = new HashMap<String, Integer>();
	static List<Integer> wordCount = new ArrayList<Integer>();
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
		wordlist = null;
		naiveBayesMatrixCreation();
	}

	public static void countDistinctWords(final File folder)
			throws IOException, FileNotFoundException,
			UnsupportedEncodingException {
		String currentLine = "";
		for (final File fileEntry : folder.listFiles()) {
			int wordsInDocument = 0;
			BufferedReader bReader = new BufferedReader(new FileReader(
					fileEntry));
			while ((currentLine = bReader.readLine()) != null) {
				currentLine = currentLine.replaceAll("[^A-Za-z ]", " ");
				currentLine = currentLine.replaceAll("\\s+", " ");
				currentLine = currentLine.toLowerCase();
				String[] temp = currentLine.split(" ");
				for (int index = 0; index < temp.length; index++) {
					if (temp[index].trim().length() > 0) {
						wordsInDocument++;
						if (!wordlist.containsKey(temp[index])) {
							wordlist.put(temp[index], globalCounter);
							globalCounter++;
						}
					}
				}
			}
			wordCount.add(wordsInDocument);
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
						if (temp[index].trim().length() > 0) {
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
		getPerplexity(posTrainingProb, negTrainingProb, testPosMatrix,
				testNegMatrix, fold);
	}

	private static void getPerplexity(double[][] posTrainingProb,
			double[][] negTrainingProb, double[][] testPosMatrix,
			double[][] testNegMatrix, int fold) {
		double posReviewToPos[][] = new double[1][200];
		double posReviewToNeg[][] = new double[1][200];
		double negReviewToPos[][] = new double[1][200];
		double negReviewToNeg[][] = new double[1][200];
		for (int i = 0; i < 200; i++) {
			for (int j = 0; j < wordlistCount; j++) {
				posReviewToPos[0][i] += testPosMatrix[i][j]
						* Math.log(posTrainingProb[0][j]);
				posReviewToNeg[0][i] += testPosMatrix[i][j]
						* Math.log(negTrainingProb[0][j]);
				negReviewToPos[0][i] += testNegMatrix[i][j]
						* Math.log(posTrainingProb[0][j]);
				negReviewToNeg[0][i] += testNegMatrix[i][j]
						* Math.log(negTrainingProb[0][j]);
			}
			posReviewToPos[0][i] = (-1 * posReviewToPos[0][i])
					/ wordCount.get(fold * 200 + i);
			posReviewToNeg[0][i] = (-1 * posReviewToNeg[0][i])
					/ wordCount.get(fold * 200 + i);
			negReviewToPos[0][i] = (-1 * negReviewToPos[0][i])
					/ wordCount.get(fold * 200 + 1000 + i);
			negReviewToNeg[0][i] = (-1 * negReviewToNeg[0][i])
					/ wordCount.get(fold * 200 + 1000 + i);

		}
		int misclasification = 0;
		for (int i = 0; i < 200; i++) {
			if (posReviewToPos[0][i] > posReviewToNeg[0][i]) {
				misclasification++;
			}
			if (negReviewToNeg[0][i] > negReviewToPos[0][i]) {
				misclasification++;
			}
		}
		System.out.println("Prediction Accuracy = " + (100 - (double) misclasification / 4));
		System.out.println("Error Rate = " + (double) misclasification / 4);
	}
}
