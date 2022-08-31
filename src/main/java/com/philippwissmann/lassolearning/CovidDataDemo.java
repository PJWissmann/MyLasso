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

		try {
			List<List<String>> table = Utilities.readCSVTableWithHeaders("C:\\Users\\phili\\eclipse-workspace\\CovidData.csv");

			countryName = table.get(0).get(country);
			
			// Read data
			for(int rowIndex = 1; rowIndex < table.size(); rowIndex++) {
				List<String> row = table.get(rowIndex);

				LocalDate date = LocalDate.parse(row.get(0), DateTimeFormatter.ISO_LOCAL_DATE);

				Double value = Double.parseDouble(row.get(country));
				values.add(value);

				System.out.println("Read " + date);
			}
			System.out.println();
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}

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

		/**
		 * Calculate grows rate
		 */
		List<Double> days = new ArrayList<>();
		List<Double> growthRates = new ArrayList<>();
		for(int i=periodLength+1; i<runningSums.size(); i++) {
			days.add((double)i);
			growthRates.add(Math.log(runningSums.get(i)/runningSums.get(i-periodLength))/periodLength);
		}
		
		Plots.createScatter(days, growthRates, 7, days.size()+7, 3)
		.setTitle("Covid Growth Rate for " + countryName)
		.setXAxisLabel("day")
		.setYAxisLabel("log growth rate")
		.show();
		
		// create the array based data set we want to use lasso on
		double[] growthRatesAsResponse = new double[growthRates.size()];
		double[][] piecewiseConstAsBasisFct = new double[growthRates.size()][growthRates.size()];
		
		for (int i = 0; i < growthRates.size(); i++) {
			growthRatesAsResponse[i] = growthRates.get(i);
			for (int j = 0; j < growthRates.size(); j++) {
				if (i>=j) {
					piecewiseConstAsBasisFct[i][j] = 1.0; // the growthRate values could be an option to use?
				} else {
					piecewiseConstAsBasisFct[i][j] = 0.0;
				}
			}
		}
		
		MyLasso lassoCovid = new MyLasso(piecewiseConstAsBasisFct, growthRatesAsResponse, false, false, false);
		
		lassoCovid.setLambda(0.0);
		lassoCovid.setMaxSteps(10000);
		lassoCovid.setTolerance(0.0);
		lassoCovid.setLearningRate(0.1);
		
		
		
	}
}

