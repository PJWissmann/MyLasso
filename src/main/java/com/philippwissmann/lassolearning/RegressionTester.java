/**
 * 
 */
package com.philippwissmann.lassolearning;


/**
 * @author phil
 *
 */
public class RegressionTester {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// some handcrafted data predictor
		double[] obs1pr ={1.2, 0.3}, obs2pr ={2.8, -0.5}, obs3pr ={-7, 0.9}, obs4pr ={3.2, -0.7}, obs5pr ={-1.2, 0.3}, obs6pr ={-3.8, -0.5}, obs7pr ={-0.2, 0.9}, obs8pr ={0.5, 0.7};
		// another handcrafted data predictor
		double[] obs1 = {1, 1, 0.3}, obs2 = {1, 2.8, -0.5}, obs3 = {1, -7, 0.9}, obs4 = {1, 3.2, -0.7};

		double[][] testPredictor = {obs1pr, obs2pr, obs3pr, obs4pr, obs5pr, obs6pr, obs7pr, obs8pr};
		double[] testResponse = {0.2+1.5, 0.6-2.5, -1.4+4.6, 0.6-3.4, -0.2+1.5, -0.8-2.5, 5, 0.1+5.5};
		
		MyLasso lassoTester = new MyLasso(testPredictor, testResponse, true, true, false); 
		
		lassoTester.setLambdaWithCV(3141, 2, 1);
//		lassoTester.setLambda(0.05);
		
		lassoTester.trainSubgradient();
		System.out.println();
		lassoTester.printBeta();
		
		lassoTester.trainCycleCoord();
		System.out.println();
		lassoTester.printBeta();
		System.out.println();
		
		lassoTester.trainGreedyCoord();
		System.out.println();
		lassoTester.printBeta();
		System.out.println();

	}

}
