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
			for (int j = 0; j < piecewiseConstAsBasisFct[0].length; j++) {
				if (i>=j) {
					piecewiseConstAsBasisFct[i][j] = 1.0; 
				} else {
					piecewiseConstAsBasisFct[i][j] = 0.0;
				}
			}
		}

		MyLasso lassoCovid = new MyLasso(piecewiseConstAsBasisFct, growthRatesAsResponse, false, false, false, true);
		
		lassoCovid.setLambda(0.0025); // 
		lassoCovid.setMaxSteps(10000);
		lassoCovid.setTolerance(0.000000001);
		lassoCovid.setLearningRate(1);
//		lassoCovid.showMeResponse(0, 2); // check the response vector
//		lassoCovid.showMeDesignMatrix(0, 2, 0, 2); // check the design matrix
		
		lassoCovid.trainGreedyCoord();		
		
		System.out.println("Centered at beta_0: " + lassoCovid.getSpecificBeta(0));

		for (int j=1; j<piecewiseConstAsBasisFct[0].length; j++) { // sparse printing
			if (lassoCovid.getSpecificBeta(j) != 0.0) {
				System.out.println("Tag " + j + " - " + dateList.get(j+21) + ": " + lassoCovid.getSpecificBeta(j));
			}
		}
		double squaredError = 0.0;
		for (int i=0; i<800; i++) {
			double temp = lassoCovid.predict(i);
			double tempError = (growthRatesAsResponse[i]-temp);
			squaredError += tempError * tempError;
//			System.out.println("actual " + growthRatesAsResponse[i] +" vs predicted " + temp + " with error " + tempError);
		}
		System.out.println("For comparison to the sklearn algorithm we computed the squared error: "+squaredError);
	}
}

