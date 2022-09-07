/**
 * For what exactly is this space used?
 */
package com.philippwissmann.lassolearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import net.finmath.lasso.Utilities;

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
	
	/**
	 * activates printing in several places that makes it easier to understand what is happening
	 */
	public boolean tellMeWhatIsHappening = false;
	
	
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
	 * boolean if the response gets centered and scaled - can be chosen in the constructor
	 * usually true so the lasso penalizes not the intercept
	 */
	public boolean responseCenteringAndStandardization;
	
	/**
	 * boolean if the features gets centered - can be chosen in the constructor
	 * usually true so the lasso penalizes not the intercept
	 */
	public boolean featureCentering;
	
	/**
	 * boolean if the features gets standardized - can be chosen in the constructor
	 * usually true so the lasso penalty hits predictors independent of their "scale"
	 */
	public boolean featureStandardization;
	
	/**
	 * boolean if the input data is already a design matrix - derived via constructor
	 * is responsible to add a "1"-column to the predictor matrix if false
	 */
    private boolean isAlreadyTheDesignMatrix;
	
	// dimensionality of the predictors - derived via constructor
	private int dimensionality;
	
	// number of observations - derived via constructor
	private int numberOfObservations;
	
	/**
	 * center of the response vector - derived via constructor
	 */
	public double centerOfTheResponse = 0;
	
	/**
	 * scaling factor of the response - derived via constructor
	 */
	public double scaleOfTheResponse = 1;
	
	/**
	 * saves the center vector of the features in the design matrix - derived via constructor if featureCentering is true
	 */
	public double[] centerVectorOfTheDesignMatrix;
	
	/**
	 * saves the scaling vector of the features in the design matrix - derived via constructor if featureStandardization is true
	 */
	public double[] scalingVectorOfTheDesignMatrix;
	
 
	
    
    
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
        this(predictor, response, false, true, true, false);
    }

    /**
     * Complete Constructor. This prepares the class before it can be trained.
     * @param predictor is the predictor matrix of the data set.
     * @param response is the response vector of the data set.
     * @param featureCentering is the boolean to center the predictor.
     * @param featureStandardization is the boolean to standardize the predictor.
     */
    public MyLasso(double[][] predictor, double[] response, boolean responseCenteringAndStandardization, boolean featureCentering, boolean featureStandardization, boolean isAlreadyTheDesignMatrix) {
    	
    	this.dimensionality = predictor[0].length; 						// derive dimensionality
        this.numberOfObservations = response.length; 					// derive number of observations
        this.responseCenteringAndStandardization = responseCenteringAndStandardization;
        this.featureCentering = featureCentering;
        this.featureStandardization = featureStandardization;
        this.isAlreadyTheDesignMatrix = isAlreadyTheDesignMatrix;


        
        // response operations
        this.centeredScaledResponse = new double[numberOfObservations];
        for (int i=0; i<numberOfObservations; i++) {												// substitute with centeredScaledResponse = response.clone()
        	centeredScaledResponse[i] = response[i];
        }
        if (responseCenteringAndStandardization) {
        	this.centerOfTheResponse = findCenter(response); 				// set the center of the response vector
        	for (int i=0; i<numberOfObservations; i++) { 					// set centered response vector
        		centeredScaledResponse[i] = centeredScaledResponse[i] - centerOfTheResponse;
        	}
        	this.scaleOfTheResponse = findStandardizationFactor(centeredScaledResponse); 	// set the scale factor of the response vector
        	for (int i=0; i<numberOfObservations; i++) { 					// set centered and scaled response vector
        		centeredScaledResponse[i] = centeredScaledResponse[i] / scaleOfTheResponse;
        	}
        }
        
        // design matrix operations - flexible if it should be centered and/or standardized
        if (this.isAlreadyTheDesignMatrix) {
        	this.designMatrix = new double[numberOfObservations][dimensionality];
        	for (int i=0; i<numberOfObservations; i++) { 				
				for (int j=0; j<dimensionality; j++) { 					
					designMatrix[i][j] = predictor[i][j];				// set designMatrix
				}
        	}
        } else {
        	dimensionality++;
        	this.designMatrix = new double[numberOfObservations][dimensionality];
        	for (int i=0; i<numberOfObservations; i++) { 				
        		designMatrix[i][0] = 1.0;								// set first column of designMatrix
				for (int j=1; j<dimensionality; j++) { 					
					designMatrix[i][j] = predictor[i][j-1];				// set other columns of designMatrix
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
        		for (int i=0; i<numberOfObservations; i++) { 			// centers the j-th feature vector of designMatrix
        			designMatrix[i][j] = designMatrix[i][j] - centerVectorOfTheDesignMatrix[j]; 
        		}
        	}
        }
        if (this.featureStandardization) { 								// if featureStandardization is true, then we standardize the feature vectors
        	this.scalingVectorOfTheDesignMatrix = new double[dimensionality];
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { 			// we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = designMatrix[i][j];																								// extract vector
        		}
        		scalingVectorOfTheDesignMatrix[j] = findStandardizationFactor(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { 			// centers the j-th feature vector of designMatrix
        			designMatrix[i][j] = designMatrix[i][j] / scalingVectorOfTheDesignMatrix[j];
        		}
        	}
        }
        
        // initialization model storage
        beta = new double[dimensionality]; 								// initialize beta vector - important note: dimensionality can change in the design matrix step!!!
        residual = new double[numberOfObservations]; 					// initialize residual vector
        
        // printing explanation
        if (tellMeWhatIsHappening) {
        	System.out.println("Variable dimensionality was set to "+ predictor[0].length + ", numberOfObservations to " + response.length +".");
        	if (responseCenteringAndStandardization) {
        		System.out.println("Vector response was centered at " + centerOfTheResponse + " and standardized with factor " + scaleOfTheResponse +".");
        	}
        	System.out.println("Matrix predictor was transformed into a design matrix if it was not already one.");
        	if (featureCentering) {				System.out.println("Also the design matrix was centered - which also removes the 1-column property of the design matrix.");		}
        	if (featureStandardization) {		System.out.println("Also the design matrix was standardized - Lasso penalizes coefficients more equally.");		}
        	System.out.println("Specify now the training parameters lambda, learningRate, tolerance and maxSteps or use the preset values.");
        	System.out.println("To train a model you can use the comfortable wrapped methods trainSubgradient(), trainCycleCoord() or trainGreedyCoord().");
        }
    }
    
    
	/* ---------- setter and getter methods for variables we use to train our lasso model ---------- */
	/**
	 * setter for the parameter lambda
	 * @param lambda as a double
	 */
	public void setLambda(double lambda) {
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
    	double[] tempError = new double[lambdaGrid.length];
    	
    	Random rng = new Random();
    	rng.setSeed(seed);
    	
    	// shuffling indexList with Collections.shuffle() to randomly assign each observation to one of the K splits
    	List<Integer> indexList = new ArrayList<>(numberOfObservations);
    	for (int i=0; i<numberOfObservations; i++) indexList.add(i%K);							
    	Collections.shuffle(indexList, rng);
    	int[] indexArrayShuffled = indexList.stream().mapToInt(Integer::intValue).toArray();
    	
    	// loop over k
    	for (int k=0; k<K; k++) {
    		// compute the testSize and trainSize
    		int testSetSize;
    		if (k >= numberOfObservations%K) {
    			testSetSize = numberOfObservations / K;
    		} else {
    			testSetSize = numberOfObservations / K + 1;	// first few chunks can have one observation more if the division leaves a rest
    		}	
    		int trainSetSize = numberOfObservations - testSetSize;
    		
    		// initialize the splitted dataset
    		double[][] cvXtrain = new double[trainSetSize][dimensionality];
    		double[] cvYtrain = new double[trainSetSize];
    		double[][] cvXtest = new double[testSetSize][dimensionality];
    		double[] cvYtest = new double[testSetSize];
    		
    		// put observations into the train or the test set depending on the current k
    		int trainSetIndex = 0, testSetIndex = 0;
    		for (int i=0; i<numberOfObservations; i++) { 	
    			if (indexArrayShuffled[i] != k) { 
    				cvXtrain[trainSetIndex] = designMatrix[i].clone();
    				cvYtrain[trainSetIndex] = centeredScaledResponse[i];
    				trainSetIndex++;
    			} else {
    				cvXtest[testSetIndex] = designMatrix[i].clone();
    				cvYtest[testSetIndex] = centeredScaledResponse[i];
    				testSetIndex++;
    			}
    		}
    		
    		// loop over lambdaGrid
    		for (int l=0; l<lambdaGrid.length; l++) {
    			if (method == 0) {
    				betaCV = trainSubgradient(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    			} else if (method == 1) {
    				betaCV = trainCycleCoord(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    			} else if (method == 2) {
    				betaCV = trainGreedyCoord(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    			}
    			tempError[l] += computeLossValue(cvXtest, cvYtest, betaCV, 0.); // we compute here the OLS loss, but this choice is flexible from a theoretical point of view 
    		}
    	}
    	
    	
    	// let's see which lambda won this and set it
    	int bestLambda = 0;
    	for (int l=0; l<lambdaGrid.length; l++) { 		
    		if (tempError[l] < tempError[bestLambda]) {
    			bestLambda = l;
    		}
    	}
    	setLambda(lambdaGrid[bestLambda]);
    	System.out.println("Lambda was set to "+ getLambda());
    	
    	
    	// printing explanation
        if (tellMeWhatIsHappening) {
        	System.out.println("The results of the " + K + "-fold cross validation within the preset lambda grid are:");
        	for (int l=0; l<lambdaGrid.length; l++) {
        		System.out.println("Lambda = " + lambdaGrid[l] + " has cumulative loss of " + tempError[l] +".");
        	}
        }
    }
	
	
	/* ---------- three different stepwise algorithms to actually train the lasso model ---------- */	
    /**
	 * This method uses the simplified assumption that the derivative for the lasso penalty is given by
	 * the sign(x) function which is one of the subgradients and then implements a gradient descent method.
	 * Goal: minarg{beta} (Y - X beta)^2 + lambda * ||beta||_1
	 * Subgradient: grad := - (Y - X beta) X + lambda * sign(beta)
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
		
		// printing explanation
		if(tellMeWhatIsHappening) System.out.println("Training via batch gradient descent in progress. Please wait...");
		if(tellMeWhatIsHappening) System.out.println(m +" "+n+" "+timeStep);
		
		// first calculate the residuals
		trainStepLoop:
		for (; timeStep <maxSteps; timeStep++) { 																		// loop over steps																				
			residualInTraining = Utilities.subtract(response, Utilities.mult(designMatrix, betaInTraining));				// compute residuals
			for (int j=0; j<n; j++) { 																						// loop over beta updates	
				double gradient;																							// compute gradient for each beta
				if (j==0) {gradient = 0.0;} else if (betaInTraining[j]<0) {gradient = lambda;} else {gradient = - lambda;}	
				for (int i=0; i<m; i++) { 
					gradient += residualInTraining[i] * designMatrix[i][j] / m;
				}
				betaUpdated[j] = betaInTraining[j] + learningRate * gradient;												// (sub-)gradient step
			}

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
		
		// printing explanation
		if(tellMeWhatIsHappening) {
			if (timeStep < maxSteps) {
				System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
			} else {
				System.out.println("You used the given computing contingent at " + maxSteps + " steps.");
			}
		}
		return betaUpdated;
	}
    
	/**
	 * This method uses the cyclic coordinate descent algorithm from the paper "COORDINATE DESCENT ALGORITHMS FOR LASSO PENALIZED REGRESSION"
	 * from WU and LANGE, The Annals of Applied Statistics, 2008.
	 * Goal: minarg{beta} (Y - X beta)^2 + lambda * ||beta||_1
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
		
		// printing explanation
		if(tellMeWhatIsHappening) System.out.println("Training via cyclic coordinate descent in progress. Please wait...");
		
		for (int i=0; i<m; i++) { 								// compute the start residuals	R = y - beta_0						// substitute with residualInTraining = response.clone()
			residualInTraining[i] = response[i];
		}
		for (int j=1; j<n; j++) { 								// compute the squardSumOfPredictors - note that we ignore j=0 since the intercept has another update formula
			for (int i=0; i<m; i++) {
				squaredSumOfJPredictors[j] += designMatrix[i][j] * designMatrix[i][j];
			}
		}
		
		trainStepLoop:
		for (; timeStep < maxSteps; timeStep++) { 													// loop over steps

			for (int j=0; j<n; j++) { 																// loop over beta updates
				if (j==0) { 									
					double interceptDerivative = 0; 												// negative sum of the residuals		
					for (int i=0; i<m; i++) { //
						interceptDerivative -= residualInTraining[i] ;	// g = - sum of R_i
					}
					betaUpdated[0] = betaInTraining[0] - /*learningRate */ interceptDerivative / m; 	// update formula for intercept
					
					for (int i=0; i<m; i++) { 					// R_i = R_i + beta_old - beta_new
						residualInTraining[i] += (betaInTraining[0]  - betaUpdated[0]);				// update the residuals
					}
				}
				else {
					double betajOLSDerivative = 0; 													// negative sum of the residuals times the X_(.j)	
					for (int i=0; i<m; i++) { //
						betajOLSDerivative -= residualInTraining[i] * designMatrix[i][j]; 
					}

					betaUpdated[j] = Math.min(0, betaInTraining[j] - (betajOLSDerivative/m - lambda)/ squaredSumOfJPredictors[j]) + 		// update formula for non-intercept
							Math.max(0, betaInTraining[j] - (betajOLSDerivative/m + lambda)/ squaredSumOfJPredictors[j]);
					for (int i=0; i<m; i++) { 					
						residualInTraining[i] += designMatrix[i][j] * (betaInTraining[j]  - betaUpdated[j]);							// update the residuals
					}
				}
			}
			
			checkMovementLoop:
			for (int j=0; j<n; j++) { 							// loop that stops the whole training if none of the beta coefficients moved more than the tolerance
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
		
		// printing explanation
		if(tellMeWhatIsHappening) {
			if (timeStep < maxSteps) {
				System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
			} else {
				System.out.println("You used the given computing contingent at " + maxSteps + " steps.");
			}
		}
		return betaUpdated;
	}

	/**
	 * This method uses the greedy coordinate descent algorithm from the paper "COORDINATE DESCENT ALGORITHMS FOR LASSO PENALIZED REGRESSION"
	 * from WU and LANGE, The Annals of Applied Statistics, 2008.
	 * Goal: minarg{beta} (Y - X beta)^2 + lambda * ||beta||_1
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
		
		// printing explanation
		if(tellMeWhatIsHappening) System.out.println("Training via greedy coordinate descent in progress. Please wait...");
		
		for (int i=0; i<m; i++) { 								// compute the start residuals					// substitute with centeredScaledResponse = response.clone()
			residualInTraining[i] = response[i];
		}
		for (int j=1; j<n; j++) { 								// compute the squaredSumOfPredictors - note this is only relevant if the data is not standardized, otherwise equal 1
			for (int i=0; i<m; i++) {
				squaredSumOfJPredictors[j] += designMatrix[i][j] * designMatrix[i][j];
			}
		}
		
		trainStepLoop:
		for (; timeStep <maxSteps; timeStep++) { 					// loop over steps

			// overhead to find the steepest derivative
			double interceptDerivative = 0; 						// first let's look at the intercept derivative
			for (int i=0; i<m; i++) { //
				interceptDerivative -= residualInTraining[i] / m;
			}
			double steepDerivative = interceptDerivative; 			// this value remembers the steepest descent, we initialize it with the intercept derivative
			int steepCoeff = 0; 									// this is the coefficient that identifies the steepest descent
			boolean isBackwardDerivative = false;
			for (int j=1; j<n; j++) { 								// search for the steepest descent - we start at j=1 because we already computed the intercept thingy
				double betajOLSDerivative = 0; 						// let's compute the derivative to compare
				for (int i=0; i<m; i++) { //
					betajOLSDerivative -= residualInTraining[i] * designMatrix[i][j] / m;
				}
				
				double forwardDerivative = betajOLSDerivative;
				if (betaInTraining[j] >= 0) { 						// here we build the directional derivatives that we want to compare depending on the sign of the coefficient ....
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
				
				if (forwardDerivative < steepDerivative) { 			// let's find out if we actually found a steeper descent
					steepDerivative = forwardDerivative;
					steepCoeff = j;
					isBackwardDerivative = false;
				} else if (backwardDerivative < steepDerivative) { 	// since our objective to minimize is convex utmost one of these conditions can be true
					steepDerivative = backwardDerivative;
					steepCoeff = j;
					isBackwardDerivative = true;
				}
			}
			
			
			if (steepDerivative >= 0) break trainStepLoop;			// now that we found the steepest descent, we check if it's really negative otherwise it won't improve beta
			
			// update step
			if (steepCoeff == 0) { 									// update the intercept
				betaUpdated[0] = betaInTraining[0] - steepDerivative ; 	
				for (int i=0; i<m; i++) { 							// update the residuals
					residualInTraining[i] += (betaInTraining[0]  - betaUpdated[0]);
				}
			} else { 												// or update another coefficient
				if (isBackwardDerivative) steepDerivative = - steepDerivative;
				betaUpdated[steepCoeff] = Math.min(0, betaInTraining[steepCoeff] - (steepDerivative)/ squaredSumOfJPredictors[steepCoeff]) + 
					Math.max(0, betaInTraining[steepCoeff] - (steepDerivative)/ squaredSumOfJPredictors[steepCoeff]);
				for (int i=0; i<m; i++) { 							// update the residuals
					residualInTraining[i] += designMatrix[i][steepCoeff] * (betaInTraining[steepCoeff]  - betaUpdated[steepCoeff]);
				}
			}
			
			if (Math.abs(betaUpdated[steepCoeff] - betaInTraining[steepCoeff]) < tolerance) {
				timeStep++;
				break trainStepLoop; 								// stops training if the update was smaller than the tolerance
			}
			
			betaInTraining = betaUpdated.clone(); 					// now to reset the trainStepLoop
		}
		
		// printing explanation
		if(tellMeWhatIsHappening) {
			if (timeStep < maxSteps) {
				System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
			} else {
				System.out.println("You used the given computing contingent at " + maxSteps + " steps.");
			}
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
	 * @return residuals as double array
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
		double yhat = Utilities.mult(x, beta);
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
		return Utilities.mult(x, beta);
	}
	
	/**
	 * Public method that uses predictors of one observations to predict a response for the saved beta vector.
	 * @param x is a double vector
	 * @return returns the predicted value
	 */
	public double predict(double[] x) {
		return predict(x, beta);
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
			lossValue += Math.pow(response[i] - predict(designMatrix[i],beta),2) / response.length;
		}
		return lossValue;
	}
	
	/**
	 * Function to print response entries.
	 * @param start index to start printing (inclusive)
	 * @param end index to stop printing (exclusive)
	 */
	public void showMeResponse(int start, int end) {
		for (int i=start; i<end; i++) {
			System.out.println(centeredScaledResponse[i]);
		}
	}
	
	/**
	 * Function to print designMatrix entries.
	 * @param startObs row index to start printing (inclusive)
	 * @param endObs row index to stop printing (exclusive)
	 * @param startJ column index to start printing (inclusive)
	 * @param endJ column index to stop printing (exclusive)
	 */
	public void showMeDesignMatrix(int startObs, int endObs, int startJ, int endJ) {
		for (int i=startObs; i<endObs; i++) {
			for (int j=startJ; j<endJ; j++) {
				System.out.print(String.format("%.4f", designMatrix[i][j])+" ");
			}
			System.out.println();
		}
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
		if (tellMeWhatIsHappening) System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
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
		if(tellMeWhatIsHappening) System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
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
		if(tellMeWhatIsHappening) System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
		updateResiduals();
	}
	

	/* ---------- self written utility functions used mainly in the constructor ---------- */
    /**
     * method to find the center of a given vector via Kahan summation
     * @param originalVector
     * @return sum(originalVector) / originalVector.length
     */
    private static double findCenter(double[] originalVector) {   	
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
     * method to find the standardization factor of a given vector via Kahan summation
     * @param originalVector
     * @return sample standard deviation
     */
    private static double findStandardizationFactor(double[] originalVector) {
    	double theTheoreticalSum = 0.0;
		double error = 0.0;
		for (int i =0; i < originalVector.length; i++) {
			double value = (double)originalVector[i] * (double)originalVector[i] - error;
			double newSum = theTheoreticalSum + value;
			error = (newSum - theTheoreticalSum) - value;
			theTheoreticalSum = newSum;
		}
		return Math.sqrt(theTheoreticalSum / (originalVector.length - 1));
    }	
}
