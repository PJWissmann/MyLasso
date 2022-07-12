/**
 * For what exactly is this space used?
 */
package com.philippwissmann.lassolearning;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Class constructed for a given dataset to train a lasso model.
 * 
 * @author Philipp Wissmann
 */
public class MyLasso {

	/*
	 * Overview to understand the code:
	 * - list of variables that describe the dataset
	 * - list of variables we use to train our lasso model
	 * - list of variables to store the result of our lasso model
	 * - basic and complete constructor
	 * - setter and getter methods for variables we use to train our lasso model
	 * - CV setter for lambda
	 * - three different stepwise algorithms to actually train the lasso model
	 * - quality of life methods
	 * - wrapper classes around the algorithms
	 * - self written utility functions
	 */
	
	/* ---------- list of variables that describe the dataset and the derived variables while the class was constructed ---------- */

	/**
	 * centered and standardized response - in most literature called Y
	 * is derived in the constructor
	 */
	public double[] centeredScaledResponse;
	
	/**
	 * design matrix version of the predictor - in most literature called X
	 * can be centered and/or scaled depending on how the class was constructed
	 * is derived in the constructor
	 */
	public double[][] designMatrix;
	
	/**
	 * boolean if the features are centered - can be chosen in the constructor
	 * usually true so the lasso penalizes not the intercept
	 */
	public boolean featureCentering = true;
	
	/**
	 * boolean if the features are standardized - can be chosen in the constructor
	 * usually true so the lasso penalty hits predictors independent of their "scale"
	 */
	public boolean featureStandardization = true;
	
	// dimensionality of the predictors - derived via constructor
	private int dimensionality;
	
	// number of observations - derived via constructor
	private int numberOfObservations;
	
	/**
	 * center of the response vector - derived via constructor
	 */
	public double centerOfTheResponse;
	
	/**
	 * scaling factor of the response - derived via constructor
	 */
	public double scaleOfTheResponse;
	
	/**
	 * saves the center vector of the features in the design matrix - derived via constructor if featureCentering is true
	 */
	public double[] centerVectorOfTheDesignMatrix;
	
	/**
	 * saves the scaling vector of the features in the design matrix - derived via constructor if featureStandardization is true
	 */
	public double[] scalingVectorOfTheDesignMatrix;
	
	/**
	 * boolean if the input data is already a design matrix - derived via constructor
	 * is responsible to add a "1"-column to the predictor matrix if false
	 */
    private boolean isAlreadyTheDesignMatrix = true;
	
    
    
    /* ---------- list of variables we use to train our lasso model ---------- */
    
	// tuning parameter lambda - pre-initialized with 0.05
	private double lambda = 0.05;
	
	// learning rate - pre-initialized with 0.01
	private double learningRate = 0.01;
	
	//tolerance - pre-initialized with 0.000001
	private double tolerance = 0.000001;

	// maximal training steps - pre-initialized with 5000
	private int maxSteps = 5000;
	
    
    
	/* ---------- list of variables to store the result of our lasso model ---------- */
	
	// linear coefficient array - this stores our trained model
	private double[] beta;
	
	// residual array - this is computed with the trained model and can be used to evaluate the model
	private double[] residual;
	
	
	
	/* ---------- basic and complete constructor ---------- */
    /**
     * Basic Constructor. This prepares the class before it can be trained.
     * @param predictor is the predictor matrix of the data set.
     * @param response is the response vector of the data set.
     */
    public MyLasso(double[][] predictor, double[] response) {
        this(predictor, response, true, true);
    }

    /**
     * Complete Constructor. This prepares the class before it can be trained.
     * @param predictor is the predictor matrix of the data set.
     * @param response is the response vector of the data set.
     * @param featureCentering is the boolean to center the predictor.
     * @param featureStandardization is the boolean to standardize the predictor.
     */
    public MyLasso(double[][] predictor, double[] response, boolean featureCentering, boolean featureStandardization) {
    	
    	this.dimensionality = predictor[0].length; 						// derive dimensionality
        this.numberOfObservations = response.length; 					// derive number of observations

        // response operations
        this.centerOfTheResponse = findCenter(response); 				// set the center of the response vector
        this.scaleOfTheResponse = findStandardizationFactor(response); 	// set the scale factor of the response vector
        this.centeredScaledResponse = new double[numberOfObservations];
        for (int i=0; i<numberOfObservations; i++) { 					// set centered and scaled response vector
        	centeredScaledResponse[i] = (response[i] - centerOfTheResponse) / scaleOfTheResponse;
        }
        
        
        // design matrix operations - flexible if the input is the design matrix or just the predictor matrix
        predictorOrDesignMatrixloop:
        for (int i=0; i<numberOfObservations; i++) { 					// if the 0-th feature of the predictor is 1 for all obs., we assume it's a design matrix
        	if (predictor[i][0] != 1) {
        		this.isAlreadyTheDesignMatrix = false;
        		break predictorOrDesignMatrixloop;
        	}
        }
        if (this.isAlreadyTheDesignMatrix) {
        	this.designMatrix = new double[numberOfObservations][dimensionality];
        	for (int i=0; i<numberOfObservations; i++) { 				// loop over observations
				for (int j=0; j<dimensionality; j++) { 					// loop over feature
					designMatrix[i][j] = predictor[i][j];
				}
        	}
        } else {
        	dimensionality++;
        	this.designMatrix = new double[numberOfObservations][dimensionality];
        	for (int i=0; i<numberOfObservations; i++) { 				// loop over observations
        		designMatrix[i][0] = 1.0;
				for (int j=1; j<dimensionality; j++) { 					// loop over feature
					designMatrix[i][j] = predictor[i][j-1];
				}
        	}
        }
        if (this.featureCentering) { 									// if featureCentering is true, then we center the feature vectors
        	this.centerVectorOfTheDesignMatrix = new double[dimensionality];
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { 			// we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = designMatrix[i][j];																								// extract vector
        		}
        		centerVectorOfTheDesignMatrix[j] = findCenter(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { 			// centers the j-th feature vector
        			designMatrix[i][j] = designMatrix[i][j] - centerVectorOfTheDesignMatrix[j];
        		}
        	}
        }
        if (this.featureStandardization) { 								// if featureStandardization is true, then we center the feature vectors
        	this.scalingVectorOfTheDesignMatrix = new double[dimensionality];
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { 			// we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = designMatrix[i][j];																								// extract vector
        		}
        		scalingVectorOfTheDesignMatrix[j] = findStandardizationFactor(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { 			// centers the j-th feature vector
        			designMatrix[i][j] = designMatrix[i][j] / scalingVectorOfTheDesignMatrix[j];
        		}
        	}
        }
        
        
        beta = new double[dimensionality]; 								// initialize beta vector
        residual = new double[numberOfObservations]; 					// initialize residual vector
        
        System.out.println("Which algorithm do you want to use to train Lasso?"); 
        System.out.println("You can use the methods trainSubgradient(), trainCycleCoord() or trainGreedyCoord()");
        System.out.println("");
        
    	// System.out.println("dimensionality "+ predictor[0].length);
        // System.out.println("numberOfObservations " + response.length);
        // System.out.println("centerResponse " + findCenter(response));
        // System.out.println("scaleResponse " + findStandardizationFactor(response));
    }
    
    
	/* ---------- setter and getter methods for variables we use to train our lasso model ---------- */
	/**
	 * setter for the parameter lambda
	 * @param lambda as a double
	 */
	public void setLambda(double lambda) {
		if (lambda < 0) {
			//throw new exception; // HERE COULD land a fitting exception
		}
		this.lambda = lambda;
	}
	
	/**
	 * getter for the parameter lambda
	 * @return current set lambda
	 */
	public double getLambda() {
		return lambda;
	}
	
	
	/**
	 * setter for the parameter learningRate
	 * @param learningRate as a double
	 */
	public void setLearningRate(double learningRate) {
		if (learningRate < 0) //throw new exception; // HERE COULD land a fitting exception
		this.learningRate = learningRate;
	}
	
	/**
	 * getter for the parameter learningRate
	 * @return current set learningRate
	 */
	public double getLearningRate() {
		return learningRate;
	}
	
	
	/**
	 * setter for the tuning parameter tolerance
	 * @param tolerance as a double
	 */
	public void setTolerance(double tolerance) {
		if (tolerance < 0) //throw new exception; // HERE COULD land a fitting exception
		this.tolerance = tolerance;
	}
	
	/**
	 * getter for the parameter tolerance
	 * @return current set tolerance
	 */
	public double getTolerance() {
		return tolerance;
	}
	
	/**
	 * setter for the tuning parameter maxStep
	 * @param maxSteps as an int 
	 */
	public void setMaxSteps(int maxSteps) {
		if (maxSteps < 0) //throw new exception; // HERE COULD land a fitting exception
		this.maxSteps = maxSteps;
	}
	/**
	 * getter for the parameter maxSteps
	 * @return current set maxSteps
	 */
	public int getMaxSteps() {
		return maxSteps;
	}
	
	
	/* ---------- CV setter for lambda ---------- */
    /**
     * Method that uses K-fold Cross Validation to find a suitable lambda from a predefined lambda grid.
     * @param seed 
     * @param K is the integer for K-fold CV
     * @param method - set to 0 for Subgradient, to 1 for CycleCoord, to 2 for GreedyCoord
     */
    public void setLambdaWithCV(int seed, int K, int method) {
    	double[] lambdaGrid = {0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0};//, 10.0, 100.0, 1000.0};
    	double[] betaCV = new double[dimensionality];
    	
    	Random rng = new Random();
    	//rng.setSeed(seed);
    	
    	// first let's shuffle our observations with Collections shuffle()
    	Integer[] indexArray = new Integer[numberOfObservations];
    	for (int i=0; i<numberOfObservations; i++) {
    		indexArray[i] = i;
    	}
    	List<Integer> indexList = Arrays.asList(indexArray);
    	Collections.shuffle(indexList, rng);
    	indexList.toArray(indexArray);
    	double[][] cvXcomplete = new double[numberOfObservations][dimensionality]; 		// initialize whole helper								// extract vector
    	double[] cvYcomplete = new double[numberOfObservations]; 						// initialize whole helper								// extract vector
    	for(int i=0; i<numberOfObservations; i++) {
    		int iShuffled = indexArray[i];
    		cvXcomplete[i] = designMatrix[iShuffled];
    		cvYcomplete[i] = centeredScaledResponse[iShuffled];
    	} // there is probably an easier way to shuffle this stuff around, but let's continue
    	
    	int kChunkSize = numberOfObservations / K;
    	int[] kChunkNumber = new int[numberOfObservations];
    	for (int i=0; i<numberOfObservations; i++) {
    		kChunkNumber[i] = i / kChunkSize;
    	}
    	
    	double[] tempError = new double[lambdaGrid.length];
    	
    	// kFoldLoop:
    	for (int k=0; k<K; k++) {
    		System.out.println("New loop number "+k+" ");
    		int testSize;
    		if (k < K-1) { 									// the last chunk could be bigger from construction
    			testSize = kChunkSize;
    		} else {
    			testSize = numberOfObservations - (K-1) * kChunkSize;
    		}
    		int trainSize = numberOfObservations - testSize;
    		double[][] cvXtrain = new double[trainSize][dimensionality];
    		double[] cvYtrain = new double[trainSize];
    		double[][] cvXtest = new double[testSize][dimensionality];
    		double[] cvYtest = new double[testSize];
    		int trainIndex = 0;
    		int testIndex = 0;
    		for (int i=0; i<numberOfObservations; i++) { 	// let's fill them
    			if (kChunkNumber[i] != k) {
    				cvXtrain[trainIndex] = cvXcomplete[i];
    				cvYtrain[trainIndex] = cvYcomplete[i];
    				trainIndex++;
    			} else {
    				cvXtest[testIndex] = cvXcomplete[i];
    				cvYtest[testIndex] = cvYcomplete[i];
    				testIndex++;
    			}
    		}
    		
    		// lambdaLoop:
    		for (int l=0; l<lambdaGrid.length; l++) {
    			// System.out.print("lambda = "+lambdaGrid[l]+" ");
    			if (method == 0) {
    				betaCV = trainSubgradient(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    			} else if (method == 1) {
    				betaCV = trainCycleCoord(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    				// System.out.println(betaCV[0]+"   "+betaCV[1]+"   "+betaCV[2]);
    			} else if (method == 2) {
    				betaCV = trainGreedyCoord(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    				// System.out.println(betaCV[0]+"   "+betaCV[1]+"   "+betaCV[2]);
    			}
    			tempError[l] += computeLossValue(cvXtest, cvYtest, betaCV, 0.); // actually we compute here the OLS loss, but this choice isn't clear from a theoretical point of view 
    			// System.out.println("tempError of lambda = "+lambdaGrid[l]+" is updated to "+ tempError[l]/(k+1));
    		}
    	}
    	
    	// let's see which lambda won this
    	int bestLambda = 0;
    	for (int l=0; l<lambdaGrid.length; l++) { 		// should start at 1
    		System.out.println("tempError of " + lambdaGrid[l] + " is " + tempError[l]);
    		if (tempError[l] < tempError[bestLambda]) {
    			bestLambda = l;
    		}
    	}

    	setLambda(lambdaGrid[bestLambda]);
    	System.out.println("Lambda was set to "+ getLambda());
    }
	
	
	/* ---------- three different stepwise algorithms to actually train the lasso model ---------- */	
    /**
	 * This method uses the simplified assumption that the derivative for the lasso penalty is given by
	 * the sign(x) function which is one of the subgradients and then implements a gradient descent method.
	 * Goal: minarg{beta} (Y - X beta)^2 + lambda ||beta||_1
	 * "gradient": grad := - (Y - X beta) X + lambda * sign(beta)
	 * one step of the gradient descent is then: beta = beta - alpha * grad
	 * @param predictor is the design matrix of our data set
	 * @param response is the response vector of our data set
	 * @param lambda is the tuning parameter
	 * @param tolerance is a small value - if no beta coefficient get's updated more than the tolerance, then the training stops
	 * @param maxSteps is maximum number of training loops we are willing to compute before the training stops
	 * @param learningRate is a factor of the gradient in the update step - low learningRate leads to better convergence, but the algorithm needs more steps to reach the goal
     * @return betaUpdated is a double vector with the trained coefficients for beta
	 */
	public double[] trainSubgradient(double[][] designMatrix, double[] response, double lambda, double tolerance, int maxSteps, double learningRate) {
		int m = response.length;
		int n = designMatrix[0].length;
		double[] betaInTraining = new double[n];
		double[] betaUpdated = new double[n];
		double[] residualInTraining = new double[m];
		int timeStep = 0;
		
		System.out.println("Training via batch gradient descent in progress. Please wait...");
		// first calculate the error
		trainStepLoop:
		for (; timeStep <maxSteps; timeStep++) { 				// loop over steps																				// product
			for (int i=0; i<m; i++) { 							// loop over errors
				residualInTraining[i] = response[i];
				for (int j=0; j<n; j++) { 
					residualInTraining[i] -= designMatrix[i][j] * betaInTraining[j];
				}
			}
			
			for (int j=0; j<n; j++) { 							// loop over beta updates																		// product
				double gradient;
				if (j==0) {gradient = 0.0;} else if (betaInTraining[j]<0) {gradient = lambda;} else {gradient = - lambda;}; 	// here is the subgradient instead of the gradient
				for (int i=0; i<m; i++) { 
					gradient += residualInTraining[i] * designMatrix[i][j];
				}
				betaUpdated[j] = betaInTraining[j] + learningRate * gradient / m;
			}
			
//			System.out.println("This is timeStep " + timeStep);
//			for (int j=0; j<dimensionality; j++) {System.out.print(betaUpdated[j] + "...");}; // testing line
//			System.out.println();

			checkMovementLoop:
			for (int j=0; j<n; j++) { 							// loop that stops the whole training if no beta coefficients moved more than the tolerance
				if (Math.abs(betaUpdated[j] - betaInTraining[j]) > tolerance) {
					break checkMovementLoop;
				}
				// System.out.println("loop number "+ j + " with " + Math.abs(betaUpdated[j] - betaInTraining[j]));
				if (j == n-1) {
					timeStep++;
					break trainStepLoop;
				}
			}
			
			betaInTraining = betaUpdated.clone(); 				// now to reset the trainStepLoop
		}
		if (timeStep < maxSteps) {
			System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
		} else {
			System.out.println("You used the given computing contingent at " + timeStep + " steps.");
		}
		return betaUpdated;
	}
    
	/**
	 * This method uses the cyclic coordinate descent algorithm from the paper "COORDINATE DESCENT ALGORITHMS FOR LASSO PENALIZED REGRESSION"
	 * from WU and LANGE, The Annals of Applied Statistics, 2008.
	 * Goal: minarg{beta} (Y - X beta)^2 + lambda ||beta||_1
	 * To reach that goal we cycle through the beta coefficients and update the coefficient analog to gradient descent but with it's forward and backward directional derivative.
     * @param predictor is the design matrix of our data set
	 * @param response is the response vector of our data set
	 * @param lambda is the tuning parameter
	 * @param tolerance is a small value - if no beta coefficient get's updated more than the tolerance, then the training stops
	 * @param maxSteps is maximum number of training loops we are willing to compute before the training stops
	 * @param learningRate is a factor of the gradient in the update step - low learningRate leads to better convergence, but the algorithm needs more steps to reach the goal
	 * @return betaUpdated is a double vector with the trained coefficients for beta
	 */
	public double[] trainCycleCoord(double[][] designMatrix, double[] response, double lambda, double tolerance, int maxSteps, double learningRate) {
		int m = response.length;
		int n = designMatrix[0].length;
		double[] betaInTraining = new double[n];
		double[] betaUpdated = new double[n];
		double[] residualInTraining = new double[m];
		double[] squaredSumOfJPredictors = new double[n];
		int timeStep = 0;
		
		System.out.println("Training via cyclic coordinate descent in progress. Please wait...");
		
		for (int i=0; i<m; i++) { 								// compute the start residuals
			residualInTraining[i] = response[i];
		}
		for (int j=1; j<n; j++) { 								// compute the squardSumOfPredictors - note that we ignore j=0 since the intercept has another update formula
			for (int i=0; i<m; i++) {
				squaredSumOfJPredictors[j] += designMatrix[i][j] * designMatrix[i][j];
			}
		}
		
		trainStepLoop:
		for (; timeStep < maxSteps; timeStep++) { 				// loop over steps

			for (int j=0; j<n; j++) { 							// loop over beta updates
				if (j==0) { 									// update the intercept
					double interceptDerivative = 0; 			// parameter that computes the negative sum of the residuals
					for (int i=0; i<m; i++) { //
						interceptDerivative -= residualInTraining[i];
					}
					betaUpdated[0] = betaInTraining[0] - learningRate / m * interceptDerivative; 
					// System.out.println("Updated beta0 from "+ betaInTraining[0] + " to " + betaUpdated[0]);
					
					for (int i=0; i<m; i++) { 					// update the residuals
						residualInTraining[i] += (betaInTraining[0]  - betaUpdated[0]);
					}
				}
				else {
					double betajOLSDerivative = 0; 				// parameter that computes the negative sum of the residuals times the x_(.j)
					
					for (int i=0; i<m; i++) { //
						betajOLSDerivative -= residualInTraining[i] * designMatrix[i][j];
					}
					betaUpdated[j] = Math.min(0, betaInTraining[j] - (betajOLSDerivative - lambda)/ squaredSumOfJPredictors[j]) + 
							Math.max(0, betaInTraining[j] - (betajOLSDerivative + lambda)/ squaredSumOfJPredictors[j]);
					// System.out.println("Updated beta"+j+" from "+ betaInTraining[j] + " to " + betaUpdated[j]);
					for (int i=0; i<m; i++) { 					// update the residuals
						residualInTraining[i] += designMatrix[i][j] * (betaInTraining[j]  - betaUpdated[j]);
					}
				}
			}

			
//			System.out.println("This is timeStep " + timeStep);
//			for (int j=0; j<dimensionality; j++) {System.out.print(betaUpdated[j] + "...");}; // testing line
//			System.out.println();
			
			checkMovementLoop:
			for (int j=0; j<n; j++) { 							// loop that stops the whole training if no beta coefficients moved more than the tolerance
				if (Math.abs(betaUpdated[j] - betaInTraining[j]) > tolerance) {
					break checkMovementLoop;
				}
				if (j == n-1) {
					timeStep++;
					break trainStepLoop;
				}
			}
			
			betaInTraining = betaUpdated.clone(); 				// now to reset the trainStepLoop
			
		}
		
		if (timeStep < maxSteps) {
			System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
		} else {
			System.out.println("You used the given computing contingent at " + timeStep + " steps.");
		}
		return betaUpdated;
	}

	/**
	 * This method uses the greedy coordinate descent algorithm from the paper "COORDINATE DESCENT ALGORITHMS FOR LASSO PENALIZED REGRESSION"
	 * from WU and LANGE, The Annals of Applied Statistics, 2008.
	 * Goal: minarg{beta} (Y - X beta)^2 + lambda ||beta||_1
	 * To reach that goal we search for the steepest descent and update this beta coefficients.
     * @param predictor is the design matrix of our data set
	 * @param response is the response vector of our data set
	 * @param lambda is the tuning parameter
	 * @param tolerance is a small value - if no beta coefficient get's updated more than the tolerance, then the training stops
	 * @param maxSteps is maximum number of training loops we are willing to compute before the training stops
	 * @param learningRate is a factor of the gradient in the update step - low learningRate leads to better convergence, but the algorithm needs more steps to reach the goal
	 * @return betaUpdated is a double vector with the trained coefficients for beta
	 */
	public double[] trainGreedyCoord(double[][] designMatrix, double[] response, double lambda, double tolerance, int maxSteps, double learningRate) {
		int m = response.length;
		int n = designMatrix[0].length;
		double[] betaInTraining = new double[n];
		double[] betaUpdated = new double[n];
		double[] residualInTraining = new double[m];
		double[] squaredSumOfJPredictors = new double[n];
		int timeStep = 0;
		
		System.out.println("Training via greedy coordinate descent in progress. Please wait...");
		
		for (int i=0; i<m; i++) { 								// compute the start residuals
			residualInTraining[i] = response[i];
		}
		for (int j=1; j<n; j++) { 								// compute the squaredSumOfPredictors - note this is only relevant if the data is not standardized, otherwise this should equal 1
			for (int i=0; i<m; i++) {
				squaredSumOfJPredictors[j] += designMatrix[i][j] * designMatrix[i][j];
			}
		}
		
		trainStepLoop:
		for (; timeStep <maxSteps; timeStep++) { 				// loop over steps

			double interceptDerivative = 0; 					// first let's look at the intercept derivative
			for (int i=0; i<m; i++) { //
				interceptDerivative -= residualInTraining[i];
			}
			double steepDerivative = interceptDerivative; 		// this value remembers the steepest descent, we initialize it with the intercept derivative
			int steepCoeff = 0; 								// this is the coefficient that identifies the steepest descent
			boolean isBackwardDerivative = false;
			// System.out.println("steepDer is " + steepDerivative + " at beta" + steepCoeff); 
			for (int j=1; j<n; j++) { 							// search for the steepest descent - we start at j=1 because we already computed the intercept thingy
				double betajOLSDerivative = 0; 					// let's compute the derivative to compare
				for (int i=0; i<m; i++) { //
					betajOLSDerivative -= residualInTraining[i] * designMatrix[i][j];
				}
				
				double forwardDerivative = betajOLSDerivative;
				if (betaInTraining[j] >= 0) { 					// here we build the directional derivatives that we want to compare depending on the sign of the coefficient ....
					forwardDerivative += lambda;
				} else {
					forwardDerivative -= lambda;
				}
				double backwardDerivative = - betajOLSDerivative;
				if (betaInTraining[j] > 0) {
					backwardDerivative -= lambda;
				} else {
					backwardDerivative += lambda;
				}
				
				if (forwardDerivative < steepDerivative) { 		// let's find out if we actually found a steeper descent
					steepDerivative = forwardDerivative;
					steepCoeff = j;
					isBackwardDerivative = false;
				} else if (backwardDerivative < steepDerivative) { 	// since our objective we want to minimize is convex utmost one of these conditions can be true
					steepDerivative = backwardDerivative;
					steepCoeff = j;
					isBackwardDerivative = true;
				}
			}
			
			// System.out.println(steepCoeff);
			// now that we found the steepest descent, we should check if it's really negative
			if (steepDerivative >= 0) break trainStepLoop;
			
			if (steepCoeff == 0) { 								// update the intercept
				betaUpdated[0] = betaInTraining[0] - learningRate / m * steepDerivative; 
				for (int i=0; i<m; i++) { 						// update the residuals
					residualInTraining[i] += (betaInTraining[0]  - betaUpdated[0]);
				}
			} else { 											// or update another coefficient
				if (isBackwardDerivative) steepDerivative = - steepDerivative;
				betaUpdated[steepCoeff] = Math.min(0, betaInTraining[steepCoeff] - (steepDerivative)/ squaredSumOfJPredictors[steepCoeff]) + 
					Math.max(0, betaInTraining[steepCoeff] - (steepDerivative)/ squaredSumOfJPredictors[steepCoeff]);
				for (int i=0; i<m; i++) { 						// update the residuals
					residualInTraining[i] += designMatrix[i][steepCoeff] * (betaInTraining[steepCoeff]  - betaUpdated[steepCoeff]);
				}
			}
			
//			System.out.println("This is timeStep " + timeStep);
//			for (int j=0; j<dimensionality; j++) {System.out.print(betaUpdated[j] + "...");}; // testing line
//			System.out.println();
			
			
			if (Math.abs(betaUpdated[steepCoeff] - betaInTraining[steepCoeff]) < tolerance) {
				timeStep++;
				break trainStepLoop; 							// stops training if the update was smaller than the tolerance
			}
			
			betaInTraining = betaUpdated.clone(); 				// now to reset the trainStepLoop
		}
		
		if (timeStep < maxSteps) {
			System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
		} else {
			System.out.println("You used the given computing contingent at " + timeStep + " steps.");
		} 
		return betaUpdated;
	}

	
	/* ---------- quality of life chapter ---------- */
	// here are methods collected to easily access the trained model (beta and residuals) as well use the model to predict a response for a new observation etc.	
	/**
	 * @return coefficient array beta 
	 */
	public double[] getBeta() {
		return beta;
	}
	/**
	 * prints each beta coefficient in a new line
	 */
	public void printBeta() {
		for (int j=0; j<dimensionality; j++) {
			System.out.println("Beta" + j + ": " + beta[j]);
		}
	}
	/**
	 * @param j specifies which element of the beta array should be returned
	 * @return beta[j]
	 */
	public double getSpecificBeta(int j) {
		return beta[j];
	}
	
	/**
	 * @return residual array
	 */
	public double[] getResiduals() {
		return residual;
	}
	/**
	 * prints each residual in a new line
	 */
	public void printResidual() {
		for (int i=0; i<numberOfObservations; i++) {
			System.out.println("Residual" + i + ": " + residual[i]);
		}
	}
	/**
	 * @param i specifies which element of the residual array should be returned
	 * @return residual[i]
	 */
	public double getSpecificResidual(int i) {
		return residual[i];
	}
	
	// private method that updates the residuals after training.
	private void updateResiduals() {
		for (int i = 0; i < numberOfObservations; i++) {
			residual[i] = centeredScaledResponse[i] - predict(designMatrix[i]);
		}
	}
	

	/**
	 * Public method that uses predictors of one observations to predict a response with the beta vector.
	 * @param x is a double vector
	 * @return returns the predicted value
	 */
	public double predictRetransformed(double[] x) {
		double yhat = 0;
		for (int j=0; j<dimensionality; j++) {
			yhat += beta[j] * x[j];
		}
		return yhat*scaleOfTheResponse + centerOfTheResponse;
	}
	
	/**
	 * Overloaded method that uses the i-th observation
	 * @param i
	 * @return predictRetransformed(designMatrix[i])
	 */
	public double predictRetranformed(int i) {
		return predictRetransformed(designMatrix[i]);
	}
	
	/**
	 * Public method that uses new observations to predict a response with the beta vector after checking if it needs to be modified.
	 * @param x is a double vector
	 * @return predictRetransformed(x)
	 */
	public double predictRetransformedNewObs(double[] x) {
		if (x.length != beta.length) 						// throw some kind of exception?
		if (featureCentering) { 							// if featureCentering is true, then we center the the new observation
        	for (int j=1; j<dimensionality; j++) {
        		x[j] = x[j] - centerVectorOfTheDesignMatrix[j];
        	}
        }
        
        if (featureStandardization) { 						// if featureCentering is true, then we center the feature vectors
        	for (int j=1; j<dimensionality; j++) {
        		x[j] = x[j] / scalingVectorOfTheDesignMatrix[j];
        	}
        }
        return predictRetransformed(x);
	}
	
	/**
	 * Public method that uses predictors of one observations to predict a response for a given beta vector.
	 * @param x is a double vector
	 * @param beta is a double vector
	 * @return returns the predicted value
	 */
	public double predict(double[] x, double[] beta) {
		double yhat = 0;
		for (int j=0; j<dimensionality; j++) {
			yhat += beta[j] * x[j];
		}
		return yhat;
	}
	
	/**
	 * Public method that uses predictors of one observations to predict a response for the saved beta vector.
	 * @param x is a double vector
	 * @return returns the predicted value
	 */
	public double predict(double[] x) {
		double[] betaTemp = this.beta.clone();
		return predict(x, betaTemp);
	}

	/**
	 * Overloaded method that uses the i-th observation
	 * @param i
	 * @return predict(designMatrix[i])
	 */
	public double predict(int i) {
		return predict(designMatrix[i]);
	}
	
	/**
	 * Computes the lasso loss.
	 * Which is the sum of the OLS loss plus lambda times the L1-norm of beta.
	 * @param designMatrix
	 * @param response
	 * @param beta
	 * @param lambda
	 * @return lasso loss as a double
	 */
	public double computeLossValue(double[][] designMatrix, double[] response, double[] beta, double lambda) {
		double betaSum = 0.;
		for(int j=0; j<beta.length; j++) {
			betaSum += beta[j];
		}
		double lossValue = lambda*betaSum;
		for (int i=0; i<response.length; i++) {
			lossValue += Math.pow(response[i] - predict(designMatrix[i], beta),2);
		}
		
		return lossValue;
	}
	
	/* ---------- wrapper classes around the algorithms that show the computation time and update the residual with the new model ---------- */
	/**
	 * Public method of the simplified gradient descent method that uses the set parameters to train.
	 * The currently set parameters are used for training. More details are given in the method with arguments.
	 */
	public void trainSubgradient() {
		long startTimeStamp = System.nanoTime();
		beta = trainSubgradient(designMatrix, centeredScaledResponse, lambda, tolerance, maxSteps, learningRate).clone();
		long endTimeStamp = System.nanoTime();
		System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
		updateResiduals();
	}
	
	/**
	 * Public method of the simplified gradient descent method that uses the set parameters to train.
	 * The currently set parameters are used for training. More details are given in the method with arguments.
	 */
	public void trainCycleCoord() {
		long startTimeStamp = System.nanoTime();
		beta = trainCycleCoord(designMatrix, centeredScaledResponse, lambda, tolerance, maxSteps, learningRate).clone();
		long endTimeStamp = System.nanoTime();
		System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
		updateResiduals();
	}
	
	/**
	 * Public method of the simplified gradient descent method that uses the set parameters to train.
	 * The currently set parameters are used for training. More details are given in the method with arguments.
	 */
	public void trainGreedyCoord() {
		long startTimeStamp = System.nanoTime();
		beta = this.trainGreedyCoord(designMatrix, centeredScaledResponse, lambda, tolerance, maxSteps, learningRate).clone();
		long endTimeStamp = System.nanoTime();
		System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
		updateResiduals();
	}
	

	/* ---------- self written utility functions used mainly in the constructor ---------- */
    /**
     * method to find the center of a given vector via Kahan summation
     * @param originalVector
     * @return sum(originalVector) / originalVector.length
     */
    private static double findCenter(double[] originalVector) {   	
    	// let's sum the response values via the Kahan sum algorithm since potential datasets can be very big
    	double theTheoreticalSum = 0.0;
		double error = 0.0;
		for (int i =0; i < originalVector.length; i++) {
			double value = (double)originalVector[i]-error;
			double newSum = theTheoreticalSum + value;
			error = (newSum - theTheoreticalSum) - value;
			theTheoreticalSum = newSum;
		}
		return theTheoreticalSum / originalVector.length;
    }  
    
    /**
     * method to find the mean squared sum of a given vector via Kahan summation
     * @param originalVector
     * @return sum(originalVector)^2 / originalVector.length
     */
    private static double findStandardizationFactor(double[] originalVector) {
    	// let's sum the squared response values via the Kahan sum algorithm since potential datasets can be very big
    	double theTheoreticalSum = 0.0;
		double error = 0.0;
		for (int i =0; i < originalVector.length; i++) {
			double value = (double)originalVector[i] * (double)originalVector[i] - error;
			double newSum = theTheoreticalSum + value;
			error = (newSum - theTheoreticalSum) - value;
			theTheoreticalSum = newSum;
		}
		return Math.sqrt(theTheoreticalSum) / originalVector.length;
    }
	

	
}
