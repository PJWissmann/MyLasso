/**
 * 
 */
package net.finmath.lasso;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * @author Christian Fries
 */
public class Utilities {

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
	
	public static double[] subArray(final double[] array, int startIndexInclusive, int endIndexExclusive) {
		return ArrayUtils.subarray(array, startIndexInclusive, endIndexExclusive);
	}
}
