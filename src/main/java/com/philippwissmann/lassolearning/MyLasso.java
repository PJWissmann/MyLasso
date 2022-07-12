/**
 * 
 */
package com.philippwissmann.lassolearning;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * @author Philipp Wissmann
 *
 */
public class MyLasso {

	// list of variables the dataset has
	
	/**
	 * dimensionality of the predictors
	 */
	private int dimensionality;
	
	/**
	 * number of observations
	 */
	private int numberOfObservations;
	
	/**
	 * center of the response
	 */
	public double centerOfTheResponse;
	
	/**
	 * scaling factor of the response
	 */
	private double scaleOfTheResponse;
	
	/**
	 * saves the centered and scaled response 
	 */
	private double[] centeredScaledResponse;
	
	/**
	 * saves the center vector of the features in the design matrix
	 */
	private double[] centerVectorOfTheDesignMatrix;
	
	/**
	 * saves the scaling vector of the features in the design matrix
	 */
	private double[] scalingVectorOfTheDesignMatrix;
	
	/**
	 * saves the design matrix of the predictor
	 */
	private double[][] designMatrix;
	
	/**
	 * saves the boolean if the features are centered
	 */
	private boolean featureCentering = true;
	
	/**
	 * saves the boolean if the features are standardized
	 */
	private boolean featureStandardization = true;
	
	/**
	 * saves the boolean if the input data is already a design matrix
	 */
    private boolean isAlreadyTheDesignMatrix = true;
	
	// list of variables where we store the result of our lasso model
	
	/**
	 * linear coefficient array with a getter method and a printer method as well as a getter for a specific value
	 */
	private double[] beta;
	public double[] getBeta() {
		return beta;
	}
	public void printBeta() {
		for (int j=0; j<dimensionality; j++) {
			System.out.println("Beta" + j + ": " + beta[j]);
		}
	}
	public double getSpecificBeta(int j) {
		return beta[j];
	}
	
	/**
	 * residual array with a getter method and a printer method as well as a getter for a specific value
	 */
	private double[] residual;
	public double[] getResiduals() {
		return residual;
	}
	public void printResidual() {
		for (int i=0; i<numberOfObservations; i++) {
			System.out.println("Residual" + i + ": " + residual[i]);
		}
	}
	public double getSpecificResidual(int i) {
		return residual[i];
	}
	
	
	//training specific parameters
	/**
	 * tuning parameter lambda
	 */
	private double lambda = 0.05;
	
	/**
	 * learning rate 
	 */
	private double learningRate = 0.01;
	
	/**
	 * tolerance with an initial freely chosen value
	 */
	private double tolerance = 0.000001;

	/**
	 * maximal training steps with an initial freely chosen value
	 */
	private int maxSteps = 5000;
	
	
	// setter and getter for training specific parameters
	/**
	 * Setter for the tuning parameter lambda.
	 * @param lambda is the double tuning parameter.
	 * @return TrainMyLasso with the new lambda.
	 */
	public void setLambda(double lambda) {
		if (lambda < 0) {
			//throw new exception; // HERE SHOULD land a fitting exception
		}
		this.lambda = lambda;
	}
	public double getLambda() {
		return lambda;
	}
	
	/**
	 * Setter for the tuning parameter learning rate.
	 * @param learningRate is the double learning rate parameter.
	 * @return TrainMyLasso with the new learning rate.
	 */
	public void setLearningRate(double learningRate) {
		if (learningRate < 0) //throw new exception; // HERE SHOULD land a fitting exception
		this.learningRate = learningRate;
	}
	public double getLearningRate() {
		return learningRate;
	}
	
	
	/**
	 * Setter for the tuning parameter tolerance.
	 * @param tolerance is the double tolerance parameter.
	 * @return TrainMyLasso with the new tolerance.
	 */
	public void setTolerance(double tolerance) {
		if (tolerance < 0) //throw new exception; // HERE SHOULD land a fitting exception
		this.tolerance = tolerance;
	}
	public double getTolerance() {
		return tolerance;
	}
	
	/**
	 * Setter for the tuning parameter maxSteps.
	 * @param maxSteps is the int maximum steps parameter.
	 * @return TrainMyLasso with the new maximum steps.
	 */
	public void setMaxSteps(int maxSteps) {
		if (maxSteps < 0) //throw new exception; // HERE SHOULD land a fitting exception
		this.maxSteps = maxSteps;
	}
	public int getMaxSteps() {
		return maxSteps;
	}
	
	/**
	 * Public method of the simplified gradient descent method that uses the set parameters to train.
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
	 */
	public void trainGreedyCoord() {
		long startTimeStamp = System.nanoTime();
		beta = this.trainGreedyCoord(designMatrix, centeredScaledResponse, lambda, tolerance, maxSteps, learningRate).clone();
		long endTimeStamp = System.nanoTime();
		System.out.println("The algorithm needed " + (endTimeStamp - startTimeStamp) / (double) 1000000 + " ms.");
		updateResiduals();
	}
	
	/**
	 * Private method that updates the residuals after training.
	 */
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
		if (x.length != beta.length) //throw some kind of exception?
		if (featureCentering) { // if featureCentering is true, then we center the the new observation
        	for (int j=1; j<dimensionality; j++) {
        		x[j] = x[j] - centerVectorOfTheDesignMatrix[j];
        	}
        }
        
        if (featureStandardization) { // if featureCentering is true, then we center the feature vectors
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
	private double[] trainSubgradient(double[][] designMatrix, double[] response, double lambda, double tolerance, int maxSteps, double learningRate) {
		int m = response.length;
		int n = designMatrix[0].length;
		double[] betaInTraining = new double[n];
		double[] betaUpdated = new double[n];
		double[] residualInTraining = new double[m];
		int timeStep = 0;
		
		System.out.println("Training via batch gradient descent in progress. Please wait...");
		// first calculate the error
		trainStepLoop:
		for (; timeStep <maxSteps; timeStep++) { // loop over steps
			for (int i=0; i<m; i++) { // loop over errors
				residualInTraining[i] = response[i];
				for (int j=0; j<n; j++) { 
					residualInTraining[i] -= designMatrix[i][j] * betaInTraining[j];
				}
			}
			
			for (int j=0; j<n; j++) { // loop over beta updates
				double gradient;
				if (j==0) {gradient = 0.0;} else if (betaInTraining[j]<0) {gradient = lambda;} else {gradient = - lambda;}; // here is the subgradient instead of the gradient
				for (int i=0; i<m; i++) { 
					gradient += residualInTraining[i] * designMatrix[i][j];
				}
				betaUpdated[j] = betaInTraining[j] + learningRate * gradient / m;
			}
			
//			System.out.println("This is timeStep " + timeStep);
//			for (int j=0; j<dimensionality; j++) {System.out.print(betaUpdated[j] + "...");}; // testing line
//			System.out.println();

			checkMovementLoop:
			for (int j=0; j<n; j++) { // loop that stops the whole training if no beta coefficients moved more than the tolerance
				if (Math.abs(betaUpdated[j] - betaInTraining[j]) > tolerance) {
					break checkMovementLoop;
				}
				// System.out.println("loop number "+ j + " with " + Math.abs(betaUpdated[j] - betaInTraining[j]));
				if (j == n-1) {
					timeStep++;
					break trainStepLoop;
				}
			}
			
			betaInTraining = betaUpdated.clone(); // now to reset the trainStepLoop
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
	private double[] trainCycleCoord(double[][] designMatrix, double[] response, double lambda, double tolerance, int maxSteps, double learningRate) {
		int m = response.length;
		int n = designMatrix[0].length;
		double[] betaInTraining = new double[n];
		double[] betaUpdated = new double[n];
		double[] residualInTraining = new double[m];
		double[] squaredSumOfJPredictors = new double[n];
		int timeStep = 0;
		
		System.out.println("Training via cyclic coordinate descent in progress. Please wait...");
		
		for (int i=0; i<m; i++) { // compute the start residuals
			residualInTraining[i] = response[i];
		}
		for (int j=1; j<n; j++) { // compute the squardSumOfPredictors - note that we ignore j=0 since the intercept has another update formula
			for (int i=0; i<m; i++) {
				squaredSumOfJPredictors[j] += designMatrix[i][j] * designMatrix[i][j];
			}
		}
		
		trainStepLoop:
		for (; timeStep < maxSteps; timeStep++) { // loop over steps

			for (int j=0; j<n; j++) { // loop over beta updates
				if (j==0) { // update the intercept
					double interceptDerivative = 0; // parameter that computes the negative sum of the residuals
					for (int i=0; i<m; i++) { //
						interceptDerivative -= residualInTraining[i];
					}
					betaUpdated[0] = betaInTraining[0] - learningRate / m * interceptDerivative; 
					// System.out.println("Updated beta0 from "+ betaInTraining[0] + " to " + betaUpdated[0]);
					
					for (int i=0; i<m; i++) { // update the residuals
						residualInTraining[i] += (betaInTraining[0]  - betaUpdated[0]);
					}
				}
				else {
					double betajOLSDerivative = 0; // parameter that computes the negative sum of the residuals times the x_(.j)
					
					for (int i=0; i<m; i++) { //
						betajOLSDerivative -= residualInTraining[i] * designMatrix[i][j];
					}
					betaUpdated[j] = Math.min(0, betaInTraining[j] - (betajOLSDerivative - lambda)/ squaredSumOfJPredictors[j]) + 
							Math.max(0, betaInTraining[j] - (betajOLSDerivative + lambda)/ squaredSumOfJPredictors[j]);
					// System.out.println("Updated beta"+j+" from "+ betaInTraining[j] + " to " + betaUpdated[j]);
					for (int i=0; i<m; i++) { // update the residuals
						residualInTraining[i] += designMatrix[i][j] * (betaInTraining[j]  - betaUpdated[j]);
					}
				}
			}

			
//			System.out.println("This is timeStep " + timeStep);
//			for (int j=0; j<dimensionality; j++) {System.out.print(betaUpdated[j] + "...");}; // testing line
//			System.out.println();
			
			checkMovementLoop:
			for (int j=0; j<n; j++) { // loop that stops the whole training if no beta coefficients moved more than the tolerance
				if (Math.abs(betaUpdated[j] - betaInTraining[j]) > tolerance) {
					break checkMovementLoop;
				}
				if (j == n-1) {
					timeStep++;
					break trainStepLoop;
				}
			}
			
			betaInTraining = betaUpdated.clone(); // now to reset the loop
			
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
	private double[] trainGreedyCoord(double[][] designMatrix, double[] response, double lambda, double tolerance, int maxSteps, double learningRate) {
		int m = response.length;
		int n = designMatrix[0].length;
		double[] betaInTraining = new double[n];
		double[] betaUpdated = new double[n];
		double[] residualInTraining = new double[m];
		double[] squaredSumOfJPredictors = new double[n];
		int timeStep = 0;
		
		System.out.println("Training via greedy coordinate descent in progress. Please wait...");
		
		for (int i=0; i<m; i++) { // compute the start residuals
			residualInTraining[i] = response[i];
		}
		for (int j=1; j<n; j++) { // compute the squaredSumOfPredictors - note this is only relevant if the data is not standardized, otherwise this should equal 1
			for (int i=0; i<m; i++) {
				squaredSumOfJPredictors[j] += designMatrix[i][j] * designMatrix[i][j];
			}
		}
		
		trainStepLoop:
		for (; timeStep <maxSteps; timeStep++) { // loop over steps

			double interceptDerivative = 0; // first let's look at the intercept derivative
			for (int i=0; i<m; i++) { //
				interceptDerivative -= residualInTraining[i];
			}
			double steepDerivative = interceptDerivative; // this value remembers the steepest descent, we initialize it with the intercept derivative
			int steepCoeff = 0; // this is the coefficient that identifies the steepest descent
			boolean isBackwardDerivative = false;
			// System.out.println("steepDer is " + steepDerivative + " at beta" + steepCoeff); 
			for (int j=1; j<n; j++) { // search for the steepest descent - we start at j=1 because we already computed the intercept thingy
				double betajOLSDerivative = 0; // let's compute the derivative to compare
				for (int i=0; i<m; i++) { //
					betajOLSDerivative -= residualInTraining[i] * designMatrix[i][j];
				}
				
				double forwardDerivative = betajOLSDerivative;
				if (betaInTraining[j] >= 0) { // here we build the directional derivatives that we want to compare depending on the sign of the coefficient ....
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
				
				if (forwardDerivative < steepDerivative) { // let's find out if we actually found a steeper descent
					steepDerivative = forwardDerivative;
					steepCoeff = j;
					isBackwardDerivative = false;
				} else if (backwardDerivative < steepDerivative) { // since our objective we want to minimize is convex utmost one of these conditions can be true
					steepDerivative = backwardDerivative;
					steepCoeff = j;
					isBackwardDerivative = true;
				}
			}
			
			// System.out.println(steepCoeff);
			// now that we found the steepest descent, we should check if it's really negative
			if (steepDerivative >= 0) break trainStepLoop;
			
			if (steepCoeff == 0) { // update the intercept
				betaUpdated[0] = betaInTraining[0] - learningRate / m * steepDerivative; 
				for (int i=0; i<m; i++) { // update the residuals
					residualInTraining[i] += (betaInTraining[0]  - betaUpdated[0]);
				}
			} else { // or update another coefficient
				if (isBackwardDerivative) steepDerivative = - steepDerivative;
				betaUpdated[steepCoeff] = Math.min(0, betaInTraining[steepCoeff] - (steepDerivative)/ squaredSumOfJPredictors[steepCoeff]) + 
					Math.max(0, betaInTraining[steepCoeff] - (steepDerivative)/ squaredSumOfJPredictors[steepCoeff]);
				for (int i=0; i<m; i++) { // update the residuals
					residualInTraining[i] += designMatrix[i][steepCoeff] * (betaInTraining[steepCoeff]  - betaUpdated[steepCoeff]);
				}
			}
			
//			System.out.println("This is timeStep " + timeStep);
//			for (int j=0; j<dimensionality; j++) {System.out.print(betaUpdated[j] + "...");}; // testing line
//			System.out.println();
			
			
			if (Math.abs(betaUpdated[steepCoeff] - betaInTraining[steepCoeff]) < tolerance) {
				timeStep++;
				break trainStepLoop; // stops training if the update was smaller than the tolerance
			}
			
			betaInTraining = betaUpdated.clone(); // now to reset the loop
		}
		
		if (timeStep < maxSteps) {
			System.out.println("You reached your destination after " + timeStep + " steps. Congrats.");
		} else {
			System.out.println("You used the given computing contingent at " + timeStep + " steps.");
		} 
		return betaUpdated;
	}
	
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
    	
    	this.dimensionality = predictor[0].length; // set the dimensionality
    	// System.out.println("dimensionality "+ predictor[0].length);
        this.numberOfObservations = response.length; // set the number of Observations
        // System.out.println("numberOfObservations " + response.length);
        
        this.centerOfTheResponse = findCenter(response); // set the center of the response vector
        System.out.println("centerResponse " + findCenter(response));
        this.scaleOfTheResponse = findStandardizationFactor(response); // set the scale factor of the response vector
        System.out.println("scaleResponse " + findStandardizationFactor(response));
        
        
        this.centeredScaledResponse = new double[numberOfObservations];
        for (int i=0; i<numberOfObservations; i++) { // let's save the centered and scaled response vector
        	centeredScaledResponse[i] = (response[i] - centerOfTheResponse) / scaleOfTheResponse;
        }
        
        
        // let's save the design matrix and be flexible if the input is the design matrix or just the predictor matrix
        predictorOrDesignMatrixloop:
        for (int i=0; i<numberOfObservations; i++) { // and let's assume that if the 0-th feature of the predictor is 1 for all observations, then it is already a design matrix
        	if (predictor[i][0] != 1) {
        		this.isAlreadyTheDesignMatrix = false;
        		break predictorOrDesignMatrixloop;
        	}
        }
        if (this.isAlreadyTheDesignMatrix) {
        	this.designMatrix = new double[numberOfObservations][dimensionality];
        	for (int i=0; i<numberOfObservations; i++) { // loop over observations
				for (int j=0; j<dimensionality; j++) { // loop over feature
					designMatrix[i][j] = predictor[i][j];
				}
        	}
        } else {
        	dimensionality++;
        	this.designMatrix = new double[numberOfObservations][dimensionality];
        	for (int i=0; i<numberOfObservations; i++) { // loop over observations
        		designMatrix[i][0] = 1.0;
				for (int j=1; j<dimensionality; j++) { // loop over feature
					designMatrix[i][j] = predictor[i][j-1];
				}
        	}
        }
        
        if (this.featureCentering) { // if featureCentering is true, then we center the feature vectors
        	this.centerVectorOfTheDesignMatrix = new double[dimensionality];
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { // we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = designMatrix[i][j];
        		}
        		centerVectorOfTheDesignMatrix[j] = findCenter(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { // centers the j-th feature vector
        			designMatrix[i][j] = designMatrix[i][j] - centerVectorOfTheDesignMatrix[j];
        		}
        	}
        	// favorite row
        }
        
        if (this.featureStandardization) { // if featureCentering is true, then we center the feature vectors
        	this.scalingVectorOfTheDesignMatrix = new double[dimensionality];
        	for (int j=1; j<dimensionality; j++) {
        		double[] helpVector = new double[numberOfObservations];
        		for (int i=0; i<numberOfObservations; i++) { // we construct a help vector because I don't know if there is a convenient way to extract the feature vectors
        			helpVector[i] = designMatrix[i][j];
        		}
        		scalingVectorOfTheDesignMatrix[j] = findStandardizationFactor(helpVector); 
        		for (int i=0; i<numberOfObservations; i++) { // centers the j-th feature vector
        			designMatrix[i][j] = designMatrix[i][j] / scalingVectorOfTheDesignMatrix[j];
        		}
        	}
        }
        
        beta = new double[dimensionality]; // initialize beta vector
        residual = new double[numberOfObservations]; // initialize residual vector
        System.out.println("Which algorithm do you want to use to train Lasso?"); 
        System.out.println("You can use the methods trainSubgradient(), trainCycleCoord() or trainGreedyCoord()");
        System.out.println("");
    }

    
    /**
     * method to find the center of a given vector via Kahan summation
     * @param originalResponse
     * @return sum(originalResponse) / originalResponse.length
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
     * @param originalResponse
     * @return sum(originalResponse) / originalResponse.length
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
    	double[][] cvXcomplete = new double[numberOfObservations][dimensionality]; // initialize whole helper
    	double[] cvYcomplete = new double[numberOfObservations]; // initialize whole helper
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
    		if (k < K-1) { // the last chunk could be bigger from construction
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
    		for (int i=0; i<numberOfObservations; i++) { // let's fill them
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
    				System.out.println(betaCV[0]+"   "+betaCV[1]+"   "+betaCV[2]);
    			} else if (method == 2) {
    				betaCV = trainGreedyCoord(cvXtrain, cvYtrain, lambdaGrid[l], tolerance, maxSteps, learningRate).clone();
    				System.out.println(betaCV[0]+"   "+betaCV[1]+"   "+betaCV[2]);
    			}
    			tempError[l] += computeLossValue(cvXtest, cvYtest, betaCV, 0.); // actually we compute here the OLS loss, but this choice isn't clear from a theoretical point of view 
    			//System.out.println("tempError of lambda = "+lambdaGrid[l]+" is updated to "+ tempError[l]/(k+1));
    		}
    	}
    	
    	// let's see which lambda won this
    	int bestLambda = 0;
    	for (int l=0; l<lambdaGrid.length; l++) { // should start at 1
    		System.out.println("tempError of " + lambdaGrid[l] + " is " + tempError[l]);
    		if (tempError[l] < tempError[bestLambda]) {
    			bestLambda = l;
    		}
    	}

    	setLambda(lambdaGrid[bestLambda]);
    	System.out.println("Lambda was set to "+ getLambda());
    }
	
}
