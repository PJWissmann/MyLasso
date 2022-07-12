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
		double[] obs1pr = {1.2, 0.3};
		double[] obs2pr = {2.8, -0.5};
		double[] obs3pr = {-7, 0.9};
		double[] obs4pr = {3.2, -0.7};
		double[] obs5pr = {-1.2, 0.3};
		double[] obs6pr = {-3.8, -0.5};
		double[] obs7pr = {-0.2, 0.9};
		double[] obs8pr = {0.5, 0.7};
		
		double[] obs1 = {1, 1, 0.3};
		double[] obs2 = {1, 2.8, -0.5};
		double[] obs3 = {1, -7, 0.9};
		double[] obs4 = {1, 3.2, -0.7};
		
		
		double[][] testPredictor = {obs1pr, obs2pr, obs3pr, obs4pr, obs5pr, obs6pr, obs7pr, obs8pr};
		double[] testResponse = {0.2+1.5, 0.6-2.5, -1.4+4.6, 0.6-3.4, -0.2+1.5, -0.8-2.5, 5, 0.1+5.5};
		
		MyLasso lassoTester = new MyLasso(testPredictor, testResponse, false, false); // true - true not yet tested
		
		
//		lassoTester.trainSubgradient();
//		System.out.println();
//		lassoTester.printBeta();
//		System.out.println();
//		lassoTester.printResidual();
//		System.out.println();
//		
//		lassoTester.trainCycleCoord();
//		System.out.println();
//		lassoTester.printBeta();
//		System.out.println();
//		
//		lassoTester.trainGreedyCoord();
//		System.out.println();
//		lassoTester.printBeta();
//		System.out.println();
//		
		lassoTester.setLambdaWithCV(0, 4, 2);
		
		lassoTester.trainCycleCoord();
		System.out.println();
		lassoTester.printBeta();
		System.out.println();
		
		lassoTester.trainGreedyCoord();
		System.out.println();
		lassoTester.printBeta();
		System.out.println();
		
		lassoTester.setLambdaWithCV(0, 4, 1);
		
		lassoTester.trainCycleCoord();
		System.out.println();
		lassoTester.printBeta();
		System.out.println();
		
		lassoTester.trainGreedyCoord();
		System.out.println();
		lassoTester.printBeta();
		System.out.println();
		lassoTester.printResidual();
		System.out.println();
		
		MyLasso lassoTester2 = new MyLasso(testPredictor, testResponse, true, true);
		
		lassoTester2.setLambdaWithCV(0, 4, 2);
		
		lassoTester2.trainCycleCoord();
		System.out.println();
		lassoTester2.printBeta();
		System.out.println();
		
		lassoTester2.trainGreedyCoord();
		System.out.println();
		lassoTester2.printBeta();
		System.out.println();
		lassoTester2.printResidual();
		System.out.println();

//		System.out.println(lassoTester.predict(obs1));
//		System.out.println(lassoTester.predict(obs2));
//		System.out.println(lassoTester.predict(obs3));
//		System.out.println(lassoTester.predict(obs4)+lassoTester.predict(obs1)+lassoTester.predict(obs2)+lassoTester.predict(obs3));
//		System.out.println(lassoTester.predictRetransformed(obs1));
//		System.out.println(lassoTester.predictRetransformed(obs2));
//		System.out.println(lassoTester.predictRetransformed(obs3));
//		System.out.println(lassoTester.predictRetransformed(obs4));
	}

}
