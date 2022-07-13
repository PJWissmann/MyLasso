/**
 * 
 */
package net.finmath.lasso;

import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * @author Christian Fries
 */
public class Utilities {

	/**
	 * Addition of two matrices.
	 *
	 * @param left The matrix A
	 * @param right The matrix B
	 * @return sum The matrix sum of A+B (if suitable)
	 */
	public static double[][] add(final double[][] left, final double[][] right) {
		return (new Array2DRowRealMatrix(left)).add(new Array2DRowRealMatrix(right)).getData();
	}
	
	/**
	 * Addition of two vectors.
	 *
	 * @param left The vector u
	 * @param right The vector v
	 * @return sum The vector sum of u+v (if suitable)
	 */
	public static double[] add(final double[] left, final double[] right) {
		return (new ArrayRealVector(left)).add(new ArrayRealVector(right)).toArray();
	}
	
	/**
	 * Subtraction of two matrices.
	 *
	 * @param left The matrix A.
	 * @param right The matrix B
	 * @return difference The matrix difference of A-B (if suitable)
	 */
	public static double[][] subtract(final double[][] left, final double[][] right) {
		return (new Array2DRowRealMatrix(left)).subtract(new Array2DRowRealMatrix(right)).getData();
	}
	
	/**
	 * Subtraction of two vectors.
	 *
	 * @param left The vector u
	 * @param right The vector v
	 * @return difference The vector difference of u-v (if suitable)
	 */
	public static double[] subtract(final double[] left, final double[] right) {
		return (new ArrayRealVector(left)).subtract(new ArrayRealVector(right)).toArray();
	}
	
	/**
	 * Multiplication of two matrices.
	 *
	 * @param left The matrix A.
	 * @param right The matrix B
	 * @return product The matrix product of A*B (if suitable)
	 */
	public static double[][] mult(final double[][] left, final double[][] right) {
		return (new Array2DRowRealMatrix(left)).multiply(new Array2DRowRealMatrix(right)).getData();
	}

	/**
	 * Multiplication of matrices and vector
	 *
	 * @param matrix The matrix A.
	 * @param vector The vector v
	 * @return product The vector product of A*v (if suitable)
	 */
	public static double[] mult(final double[][] matrix, final double[] vector) {
		return (new Array2DRowRealMatrix(matrix)).operate(vector);
	}
	
	/**
	 * Multiplication of two vectors
	 *
	 * @param left The vector u
	 * @param right The vector v
	 * @return product The dot product of u*v (if suitable)
	 */
	public static double mult(final double[] left, final double[] right) {
		return (new ArrayRealVector(left)).dotProduct(new ArrayRealVector(right));
	}
	
	/**
	 * Mapped vector
	 * 
	 * @param vector The vector u
	 * @param function The univariate function f
	 * @return mappedValue The mapped vector f(u)
	 */
	public static double[] map(final double[] vector, final UnivariateFunction function) {
		return (new ArrayRealVector(vector)).map(function).toArray();
	}
	
	/**
	 * 
	 * @param array
	 * @param startIndexInclusive
	 * @param endIndexExclusive
	 * @return subArray 
	 */
	public static double[] subArray(final double[] array, int startIndexInclusive, int endIndexExclusive) {
		return ArrayUtils.subarray(array, startIndexInclusive, endIndexExclusive);
	}
	
	/**
	 * 
	 * @param array
	 * @param startIndexInclusive
	 * @param endIndexExclusive
	 * @return subArray 
	 */
	public static <T> T[] subArray(final T[] array, int startIndexInclusive, int endIndexExclusive) {
		return ArrayUtils.subarray(array, startIndexInclusive, endIndexExclusive);
	}
	
	/**
	 * 
	 * @param array
	 * @param random
	 */
	public static void shuffle(final Object[] array, final Random random) {
		ArrayUtils.shuffle(array, random);
	}
	
}
