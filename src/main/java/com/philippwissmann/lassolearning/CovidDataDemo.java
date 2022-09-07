/**
 * 
 */
package com.philippwissmann.lassolearning;

import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import net.finmath.lasso.Utilities;
import net.finmath.plots.Plots;

/**
 * @author fries, wissmann
 *
 */
public class CovidDataDemo {

	/**
	 * 1: BW, 2: BY, ..., 7: HE, ..., 10: NRW, ... 16: TH, 17: DE (all)
	 */
	private static final int country = 17;
	private static final int periodLength = 14;

	public static void main(String[] args) {

		String countryName;
		List<Double> values = new ArrayList<>();
		List<LocalDate> dateList = new ArrayList<>();

		try {
			List<List<String>> table = Utilities.readCSVTableWithHeaders("C:\\Users\\phili\\eclipse-workspace\\CovidData.csv");

			countryName = table.get(0).get(country);
			
			// Read data
			for(int rowIndex = 1; rowIndex < table.size(); rowIndex++) {
				List<String> row = table.get(rowIndex);

				LocalDate date = LocalDate.parse(row.get(0), DateTimeFormatter.ISO_LOCAL_DATE);
				dateList.add(date);

				Double value = Double.parseDouble(row.get(country));
				values.add(value);

//				System.out.println("Read " + date);
			}
			System.out.println("Completed reading all dates.");
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
//		System.out.println(values.size());

		/**
		 * Calculate running sum over 7 days.
		 */
		List<Double> runningSums = new ArrayList<>();
		double runningSum = 0;
		for(int i=0; i<7; i++) {
			runningSum += values.get(i);
		}
		runningSums.add(runningSum);

		for(int i=7; i<values.size(); i++) {
			runningSum += values.get(i)-values.get(i-7);
			runningSums.add(runningSum);
		}
//		System.out.println(runningSums.size());

		/**
		 * Calculate grows rate
		 */
		List<Double> days = new ArrayList<>();
		List<Double> growthRates = new ArrayList<>();
		for(int i=periodLength+1; i<runningSums.size(); i++) {
			days.add((double)i);
			growthRates.add(Math.log(runningSums.get(i)/runningSums.get(i-periodLength))/periodLength);
		}
//		System.out.println(growthRates.size());
		
		
//		Plots.createScatter(days, growthRates, 7, days.size()+7, 3)
//		.setTitle("Covid Growth Rate for " + countryName)
//		.setXAxisLabel("day")
//		.setYAxisLabel("log growth rate")
//		.show();
		
		// create the array based data set we want to use lasso on
		double[] growthRatesAsResponse = new double[growthRates.size()];
		double[][] piecewiseConstAsBasisFct = new double[growthRates.size()][growthRates.size()];
		
		for (int i = 0; i < growthRates.size(); i++) {
			growthRatesAsResponse[i] = growthRates.get(i);
//			if (i<10||i>(growthRates.size()-10)) {System.out.print(growthRatesAsResponse[i] + " ");}
			for (int j = 0; j < piecewiseConstAsBasisFct[0].length; j++) {
				if (i>=j) {
					piecewiseConstAsBasisFct[i][j] = 1.0; // the growthRate values could be an option to use?
				} else {
					piecewiseConstAsBasisFct[i][j] = 0.0;
				}
//				if (i<10||i>(growthRates.size()-10)) {System.out.print(piecewiseConstAsBasisFct[i][j]+" ");}
			}
//			if (i<10||i>(growthRates.size()-10)) {System.out.println();}
		}

		MyLasso lassoCovid = new MyLasso(piecewiseConstAsBasisFct, growthRatesAsResponse, false, true, true, true);
		
//		lassoCovid.setLambdaWithCV(3141, 5, 1);
//		lassoCovid.setLambda(0.17); // if centered and scaled response 0.15 to 0.17 gives interesting results
		lassoCovid.setLambda(0.006); // if raw response 0.006 gives interesting results
		lassoCovid.setMaxSteps(100);
		lassoCovid.setTolerance(0.0);
		lassoCovid.setLearningRate(1);
		
		lassoCovid.trainCycleCoord();
		
		double[] resultBeta = new double[piecewiseConstAsBasisFct[0].length];
		resultBeta = lassoCovid.getBeta();
		
		System.out.println("Centered at beta_0: " + resultBeta[0]);
		lassoCovid.showMeResponse(0, 2);
		lassoCovid.showMeDesignMatrix(0, 5, 0, 5);
		for (int j=1; j<piecewiseConstAsBasisFct[0].length; j++) { // carefull with uncentered response beta0 takes the center value
			if (resultBeta[j] != 0.0) {
				System.out.println("Tag " + j + " - " + dateList.get(j-1) + ": " + String.format("%.14f", resultBeta[j]));
			}
		}
		
	}
}

