/**
 * 
 */
package com.philippwissmann.lassolearning;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;

import org.apache.commons.lang3.StringUtils;

/**
 * @author Philipp Wissmann
 *
 */
public class BostonHousingDemo {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		// --------------------------------- read in some data ---------------------------------
		double[][] bostonPredictor = new double[506][13];
		double[] bostonResponse = new double[506];
		
		String path = "C:\\Users\\phili\\eclipse-workspace\\Boston.csv";
		String line = "";
		int rowIndex = 0;
		try {
			BufferedReader bostonReader = new BufferedReader(new FileReader(path));
			
			while((line = bostonReader.readLine()) != null) {
				String[] valuesInCSV = line.split(",");
				if (rowIndex ==0) {
					System.out.println("Start reading the CSV.");
				} else {
					for (int j=1; j<14; j++) {
						bostonPredictor[rowIndex-1][j-1] =  Double.parseDouble(valuesInCSV[j]);
					}
					bostonResponse[rowIndex-1] = Double.parseDouble(valuesInCSV[14]);
				}
				rowIndex++;
			}
			System.out.println("Finished reading the CSV.");
			bostonReader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch(IOException e) {
			e.printStackTrace();
		}
		
		// --------------------------------- prepare some arrays to save results ---------------------------------
		double[] betaOLS = new double[14];
		double[] betaLassoSubgradient = new double[14];
		double[] betaLassoCycleCoord = new double[14];
		double[] betaLassoGreedyCoord = new double[14];
		double l1SumOLS=0, l1SumSub=0, l1SumCyc=0, l1SumGrd=0;
		
		
		// --------------------------------- construct the class on top of the data ---------------------------------
		MyLasso bostonLasso = new MyLasso(bostonPredictor, bostonResponse, false, true, true, false); 
		
		
		// --------------------------------- set training relevant parameters for an OLS solution (if we don't want to use the preset ones) ---------------------------------
		bostonLasso.setLambda(0.0);
		bostonLasso.setMaxSteps(10000);
		bostonLasso.setTolerance(0.000001);
		bostonLasso.setLearningRate(0.01);
		
		// --------------------------------- compute OLS solution ---------------------------------
		bostonLasso.trainSubgradient();
		betaOLS = bostonLasso.getBeta();
		
		// --------------------------------- compute Lasso solution for lambda = 0.5  ---------------------------------
		bostonLasso.setLambda(0.5);
		
		bostonLasso.trainSubgradient();
		betaLassoSubgradient = bostonLasso.getBeta().clone();
		
		bostonLasso.trainCycleCoord();
		betaLassoCycleCoord = bostonLasso.getBeta().clone();
		
		bostonLasso.trainGreedyCoord();
		betaLassoGreedyCoord = bostonLasso.getBeta().clone();
		
		// --------------------------------- demonstration of 5-fold CV  ---------------------------------
		System.out.println("5-fold CV would suggest to set following lambda:");
		bostonLasso.setLambdaWithCV(300, 5, 1);
		
		// --------------------------------- some printing  ---------------------------------
		System.out.println("For lambda=0.5 we get following results:");
		System.out.println("lambda=0.5     OLS solution             Lasso via subgradient    Lasso via cycleCoord     Lasso via GreedyCoord");
		for (int j=0; j<betaOLS.length; j++) {
			System.out.printf("%-15s","Beta" + j);
			System.out.printf("%-25s", betaOLS[j]);
			System.out.printf("%-25s", betaLassoSubgradient[j]);
			System.out.printf("%-25s", betaLassoCycleCoord[j]);
			System.out.println(betaLassoGreedyCoord[j]);
		}
		for (int j=1; j<betaOLS.length; j++) {
			l1SumOLS += Math.abs(betaOLS[j]);
			l1SumSub += Math.abs(betaLassoSubgradient[j]);
			l1SumCyc += Math.abs(betaLassoCycleCoord[j]);
			l1SumGrd += Math.abs(betaLassoGreedyCoord[j]);
		}
		System.out.printf("%-15s","AbsSum of 1-13");
		System.out.printf("%-25s", l1SumOLS);
		System.out.printf("%-25s", l1SumSub);
		System.out.printf("%-25s", l1SumCyc);
		System.out.println(l1SumGrd);
		
		System.out.printf("%-15s","OLS loss");
		System.out.printf("%-25s", bostonLasso.computeLossValue(bostonLasso.designMatrix, bostonLasso.centeredScaledResponse, betaOLS, 0.0));
		System.out.printf("%-25s", bostonLasso.computeLossValue(bostonLasso.designMatrix, bostonLasso.centeredScaledResponse, betaLassoSubgradient, 0.0));
		System.out.printf("%-25s", bostonLasso.computeLossValue(bostonLasso.designMatrix, bostonLasso.centeredScaledResponse, betaLassoCycleCoord, 0.0));
		System.out.println(bostonLasso.computeLossValue(bostonLasso.designMatrix, bostonLasso.centeredScaledResponse, betaLassoGreedyCoord, 0.0));
	}

}
