package matrix

// http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html
import (
	"fmt"
	"math/cmplx"
	"sort"
)

// Private
// Get a column in the form of a slice from a matrix
func column(matrix [][]float64, columnNumber int) (result []float64) {
	result = make([]float64, len(matrix))
	for ii := 0; ii < len(matrix); ii++ {
		for jj := 0; jj < len(matrix[0]); jj++ {
			result[ii] = matrix[ii][columnNumber]
		}
	}
	return
}

// Returns the sum of the given two matrix
func Add(firstMatrix [][]float64, secondMatrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(firstMatrix))
	for ii := 0; ii < len(firstMatrix); ii++ {
		result[ii] = make([]float64, len(firstMatrix[0]))
		for jj := 0; jj < len(firstMatrix[0]); jj++ {
			result[ii][jj] = firstMatrix[ii][jj] + secondMatrix[ii][jj]
		}
	}
	return
}

// Subtraction operation on two matrices
func Subtract(firstMatrix [][]float64, secondMatrix [][]float64) [][]float64 {
	secondMatrix = MultiplyByScalar(secondMatrix, -1.0)
	return Add(firstMatrix, secondMatrix)
}

// Dot (Inner) product
func DotProduct(firstMatrix [][]float64, secondMatrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(firstMatrix))
	for ii := 0; ii < len(firstMatrix); ii++ {
		result[ii] = make([]float64, len(secondMatrix[0]))
		for jj := 0; jj < len(secondMatrix[0]); jj++ {
			for kk := 0; kk < len(secondMatrix); kk++ {
				result[ii][jj] += firstMatrix[ii][kk] * secondMatrix[kk][jj]
			}
		}
	}
	return
}

// Calculates the determinant of the matrix
func Determinant(matrix [][]float64) (result float64) {
	matrixLength := len(matrix)
	sums := make([]float64, matrixLength*2)
	for ii := 0; ii < len(sums); ii++ {
		sums[ii] = 1
	}

	for ii := 0; ii < matrixLength; ii++ {
		for jj := 0; jj < matrixLength; jj++ {
			if ii-jj < 0 {
				sums[matrixLength+ii-jj] *= matrix[ii][jj]
			} else {
				sums[ii-jj] *= matrix[ii][jj]
			}

			if ii+jj >= matrixLength {
				sums[ii+jj] *= matrix[ii][jj]
			} else {
				sums[ii+jj+matrixLength] *= matrix[ii][jj]
			}
		}
	}

	dim := matrixLength * 2
	if matrixLength == 2 {
		dim = 2
		matrixLength = 1
	}

	for ii := 0; ii < dim; ii++ {
		if ii >= matrixLength {
			result -= sums[ii]
		} else {
			result += sums[ii]
		}
	}
	return
}

// Minor matrix of a given matrix
func MinorMatrix(matrix [][]float64) (result [][]float64) {
	var (
		matrixLength int
	)

	matrixLength = len(matrix)
	result = make([][]float64, matrixLength)
	for ii := 0; ii < matrixLength; ii++ {
		result[ii] = make([]float64, matrixLength)
		for jj := 0; jj < matrixLength; jj++ {
			auxM := [][]float64{}
			for iik := 0; iik < matrixLength; iik++ {
				if iik != ii {
					auxR := []float64{}
					for jjk := 0; jjk < matrixLength; jjk++ {
						if jjk != jj {
							auxR = append(auxR, matrix[iik][jjk])
						}
					}
					auxM = append(auxM, auxR)
				}
			}
			result[ii][jj] = Determinant(auxM)
		}
	}
	return
}

// Returns the Cofactor Matrix
func CofactorMatrix(matrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(matrix))
	for ii := 0; ii < len(matrix); ii++ {
		result[ii] = make([]float64, len(matrix[0]))
		for jj := 0; jj < len(matrix[0]); jj++ {
			if (ii+jj)%2 == 0 {
				result[ii][jj] = matrix[ii][jj]
			} else {
				result[ii][jj] = -matrix[ii][jj]
			}
		}
	}
	return
}

// Calculates the inverse matrix
func Inverse(matrix [][]float64) [][]float64 {
	determinant := Determinant(matrix)
	adj := Transpose(CofactorMatrix(MinorMatrix(matrix)))
	return MultiplyByScalar(adj, 1./determinant)
}

// Divide the first matrix by the second one
func Divide(firstMatrix [][]float64, secondMatrix [][]float64) [][]float64 {
	return DotProduct(firstMatrix, Inverse(secondMatrix))
}

// Returns the rm of multiply all the elements of a matrix by a float number
func MultiplyByScalar(matrix [][]float64, scalar float64) [][]float64 {
	function := func(x float64) float64 {
		return x * scalar
	}
	return Apply(matrix, function)
}

// Multiply on matrix by the Transposepose of the second matrix
func MultTranspose(firstMatrix [][]float64, secondMatrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(firstMatrix))
	for ii := 0; ii < len(firstMatrix); ii++ {
		result[ii] = make([]float64, len(secondMatrix))
		for jj := 0; jj < len(secondMatrix); jj++ {
			for kk := 0; kk < len(secondMatrix[0]); kk++ {
				result[ii][jj] += firstMatrix[ii][kk] * secondMatrix[jj][kk]
			}
		}
	}
	return
}

// Multiplication of two matrices; element-wise
func Multiply(firstMatrix [][]float64, secondMatrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(firstMatrix))
	for ii := 0; ii < len(firstMatrix); ii++ {
		result[ii] = make([]float64, len(firstMatrix[0]))
		for jj := 0; jj < len(firstMatrix[0]); jj++ {
			result[ii][jj] = firstMatrix[ii][jj] * secondMatrix[ii][jj]
		}
	}
	return
}

// Matrix Transpose
func Transpose(matrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(matrix[0]))
	// Initialize the matrix
	for ii := 0; ii < len(matrix[0]); ii++ {
		result[ii] = make([]float64, len(matrix))
		for jj := 0; jj < len(matrix); jj++ {
			result[ii][jj] = matrix[jj][ii]
		}
	}
	return
}

// Sum of the matrix along with axis
// axis=0 => row-wise
// axis=1 => column-wise
func Sum(matrix [][]float64, axis int) (result []float64, ok bool) {
	rowSum := make([]float64, len(matrix))
	columnSum := make([]float64, len(matrix[0]))
	for ii := 0; ii < len(matrix); ii++ {
		for jj := 0; jj < len(matrix[0]); jj++ {
			rowSum[ii] += matrix[ii][jj]
			columnSum[jj] += matrix[ii][jj]
		}
	}
	if axis == 1 {
		result = rowSum
		ok = true
	} else {
		result = columnSum
		ok = true
	}
	return
}

// Maximum of matrix along with axis
// axis=0 => row-wise
// axis=1 => column-wise
func Max(matrix [][]float64, axis int) (result []float64, ok bool) {
	var (
		firstDimension  int
		secondDimension int
	)
	firstDimension = len(matrix)
	secondDimension = len(matrix[0])

	rowMax := make([]float64, firstDimension)
	columnMax := make([]float64, secondDimension)
	deepMatrix := deepCopyMatrix(matrix)

	for ii := 0; ii < firstDimension; ii++ {
		sort.Float64s(deepMatrix[ii])
		rowMax[ii] = deepMatrix[ii][secondDimension-1]

		if ii == firstDimension-1 {
			for jj := 0; jj < secondDimension; jj++ {
				deepCol := deepCopyArray(column(matrix, jj))
				sort.Float64s(deepCol)
				columnMax[jj] = deepCol[len(deepCol)-1]
			}
		}
	}
	if axis == 1 {
		result = rowMax
		ok = true
	} else {
		result = columnMax
		ok = true
	}
	return
}

// Minimum of matrix along with axis
// axis=0 => row-wise
// axis=1 => column-wise
func Min(matrix [][]float64, axis int) (result []float64, ok bool) {
	var (
		firstDimension  int
		secondDimension int
	)
	firstDimension = len(matrix)
	secondDimension = len(matrix[0])

	rowMin := make([]float64, firstDimension)
	columnMin := make([]float64, secondDimension)
	deepMatrix := deepCopyMatrix(matrix)

	for ii := 0; ii < firstDimension; ii++ {
		sort.Float64s(deepMatrix[ii])
		rowMin[ii] = deepMatrix[ii][0]

		if ii == firstDimension-1 {
			for jj := 0; jj < secondDimension; jj++ {
				deepCol := deepCopyArray(column(matrix, jj))
				sort.Float64s(deepCol)
				columnMin[jj] = deepCol[0]
			}
		}
	}
	if axis == 1 {
		result = rowMin
		ok = true
	} else {
		result = columnMin
		ok = true
	}
	return
}

// Median of matrix along with axis
// axis=0 => row-wise
// axis=1 => column-wise
func Median(matrix [][]float64, axis int) (result []float64, ok bool) {
	var (
		firstDimension       int
		secondDimension      int
		halfRow              int
		halfCol              int
		isFirstDivisibleBy2  bool
		isSecondDivisibleBy2 bool
	)
	firstDimension = len(matrix)
	secondDimension = len(matrix[0])

	if firstDimension%2 == 0 {
		halfRow = firstDimension / 2
		isFirstDivisibleBy2 = true
	} else {
		halfRow = (firstDimension - 1) / 2
	}

	if secondDimension%2 == 0 {
		halfCol = secondDimension / 2
		isSecondDivisibleBy2 = true
	} else {
		halfCol = (secondDimension - 1) / 2
	}
	rowMedian := make([]float64, firstDimension)
	columnMedian := make([]float64, secondDimension)

	deepMatrix := deepCopyMatrix(matrix)

	for ii := 0; ii < firstDimension; ii++ {
		sort.Float64s(deepMatrix[ii])
		if isSecondDivisibleBy2 && secondDimension > halfRow+1 {
			rowMedian[ii] = (deepMatrix[ii][halfRow] + deepMatrix[ii][halfRow+1]) / 2.
		} else if secondDimension == 2 {
			rowMedian[ii] = (deepMatrix[ii][0] + deepMatrix[ii][1]) / 2.
		} else {
			rowMedian[ii] = deepMatrix[ii][halfRow]
		}

		if ii == firstDimension-1 {
			for jj := 0; jj < secondDimension; jj++ {
				deepCol := deepCopyArray(column(matrix, jj))
				sort.Float64s(deepCol)
				if isFirstDivisibleBy2 && len(deepCol) > halfCol+1 {
					columnMedian[jj] = (deepCol[halfCol] + deepCol[halfCol+1]) / 2.
				} else if len(deepCol) == 2 {
					columnMedian[jj] = (deepCol[0] + deepCol[1]) / 2.
				} else {
					columnMedian[jj] = deepCol[halfCol]
				}

			}
		}
	}

	if axis == 1 {
		result = rowMedian
		ok = true
	} else {
		result = columnMedian
		ok = true
	}
	return
}

// Mean of matrix along with axis
// axis=0 => row-wise
// axis=1 => column-wise
func Mean(matrix [][]float64, axis int) (result []float64, ok bool) {
	var dim int
	sum, ok := Sum(matrix, axis)
	if !ok {
		return
	}
	result = make([]float64, len(sum))
	if axis == 0 {
		dim = len(matrix)
	} else {
		dim = len(matrix[0])
	}
	for ii, value := range sum {
		result[ii] = value / float64(dim)
	}
	return
}

// Cumulative Sum of Matrix along with axis
// axis=0 => row-wise
// axis=1 => column-wise
func CumulativeSum(matrix [][]float64, axis int) (result []float64, ok bool) {
	result, ok = Sum(matrix, axis)
	temp := 0.
	for ii, jj := range result {
		temp += jj
		result[ii] = temp
	}
	return
}

// Sum all the elements in a matrix
func SumAll(m [][]float64) (result float64) {
	for ii := 0; ii < len(m); ii++ {
		for jj := 0; jj < len(m[0]); jj++ {
			result += m[ii][jj]
		}
	}
	return
}

// Apply a function to all the elements of a matrix,
//the function will receive a  float64 as param and returns a float64
func Apply(matrix [][]float64, function func(x float64) float64) (result [][]float64) {
	result = make([][]float64, len(matrix))
	for ii := 0; ii < len(matrix); ii++ {
		result[ii] = make([]float64, len(matrix[0]))
		for jj := 0; jj < len(matrix[0]); jj++ {
			result[ii][jj] = function(matrix[ii][jj])
		}
	}
	return
}

// Apply a function to a complex matrix
// the function will receive a complex128 and returns a complex128
func ComplexApply(matrix [][]complex128, function func(x complex128) complex128) (result [][]complex128) {
	result = make([][]complex128, len(matrix))
	for ii := 0; ii < len(matrix); ii++ {
		result[ii] = make([]complex128, len(matrix[0]))
		for jj := 0; jj < len(matrix[0]); jj++ {
			result[ii][jj] = function(matrix[ii][jj])
		}
	}
	return
}

// Deep Copy of an Array
func deepCopyMatrix(matrix [][]float64) (deepCopy [][]float64) {
	deepCopy = make([][]float64, len(matrix))
	for ii := 0; ii < len(matrix); ii++ {
		deepCopy[ii] = make([]float64, len(matrix[ii]))
		for jj := 0; jj < len(matrix[ii]); jj++ {
			deepCopy[ii][jj] = matrix[ii][jj]
		}
	}
	return
}

func deepCopyArray(array []float64) (deepCopy []float64) {
	deepCopy = make([]float64, len(array))
	for ii := 0; ii < len(array); ii++ {
		deepCopy[ii] = array[ii]
	}
	return
}

// Concatenate two matrices along with their axises
// axis=0 => row-wise
// axis=1 => column-wise
func Concatenate(firstMatrix [][]float64, secondMatrix [][]float64, axis int) (result [][]float64) {
	if axis == 0 {
		result = make([][]float64, len(firstMatrix)+len(secondMatrix))
		for ii := 0; ii < len(firstMatrix)+len(secondMatrix); ii++ {
			if ii < len(firstMatrix) {
				result[ii] = firstMatrix[ii]
			} else {
				result[ii] = secondMatrix[ii-len(firstMatrix)]
			}
		}
	} else {
		result = make([][]float64, len(firstMatrix))
		for i := 0; i < len(firstMatrix); i++ {
			result[i] = make([]float64, len(firstMatrix[i])+len(secondMatrix[i]))
			for j := 0; j < len(firstMatrix[i]); j++ {
				result[i][j] = firstMatrix[i][j]
			}
			for j := 0; j < len(secondMatrix[i]); j++ {
				result[i][j+len(firstMatrix[i])] = secondMatrix[i][j]
			}
		}
	}
	return
}

// Returns an array where diagonal elements are 1 and remaining
// positions are 0
// If it has been passed one parameter, it yields a square matrix
// Two parameters define the size of the matrix
func Eye(args ...int) (result [][]float64, ok bool) {
	result, ok = Zeros(args)
	if ok {
		for ii := 0; ii < len(result); ii++ {
			for jj := 0; jj < len(result[0]); jj++ {
				if ii == jj {
					result[ii][jj] = 1.
				}
			}
		}
	}
	return
}

// Returns an array filled with 0.s
// If it has been passed one parameter, it yields a square matrix
// Two parameters define the size of the matrix
func Zeros(args []int) (result [][]float64, ok bool) {
	dims := make([]int, 2)
	var isValid bool
	if len(args) == 1 {
		dims[0] = args[0]
		dims[1] = args[0]
		isValid = true
	} else if len(args) == 2 {
		dims[0] = args[0]
		dims[1] = args[1]
		isValid = true
	}
	if isValid {
		result = make([][]float64, dims[0])
		for ii := 0; ii < dims[0]; ii++ {
			result[ii] = make([]float64, dims[1])
		}
		ok = true
	}
	return
}

// Returns the diagonal of a matrix
func Diagonal(matrix [][]float64) (result []float64, ok bool) {
	var (
		minDimension int
		ii           int
		jj           int
	)
	firstDimension, secondDimension := len(matrix), len(matrix[0])
	if firstDimension < secondDimension {
		minDimension = firstDimension
	} else {
		minDimension = secondDimension
	}
	for ii = 0; ii < minDimension; ii++ {
		for jj = 0; jj < minDimension; jj++ {
			if ii == jj {
				result = append(result, matrix[ii][jj])
			}
		}
		ok = true
	}
	return
}

// Lower Triangle Matrix
func LowerTriangle(matrix [][]float64) (result [][]float64) {
	result = make([][]float64, len(matrix))
	for ii := 0; ii < len(matrix); ii++ {
		result[ii] = make([]float64, len(matrix[ii]))
		for jj := 0; jj < len(matrix[0]); jj++ {
			if ii >= jj {
				result[ii][jj] = matrix[ii][jj]
			}

		}
	}
	return
}

// Upper Triangle of Matrix
func UpperTriangle(matrix [][]float64) (result [][]float64) {
	firstDimension, secondDimension := len(matrix), len(matrix[0])
	result = make([][]float64, firstDimension)
	for ii := 0; ii < firstDimension; ii++ {
		result[ii] = make([]float64, secondDimension)
		for jj := 0; jj < secondDimension; jj++ {
			if ii <= jj {
				result[ii][jj] = matrix[ii][jj]
			}
		}
	}
	return
}

// Take the elements of the matrix given in indices
// uses so called fancy indexing to determine the positions of the
// array
func Take(matrix [][]float64, indices []int) (result []float64, ok bool) {
	var (
		first  int
		second int
	)
	firstDimension, secondDimension := len(matrix), len(matrix[0])
	sort.Ints(indices)
	if indices[len(indices)-1] < firstDimension*secondDimension {
		ok = true
	}
	if ok {
		for ii := 0; ii < len(indices); ii++ {
			first, second = indices[ii]/secondDimension, indices[ii]%secondDimension
			result = append(result, matrix[first][second])
		}
	}

	return
}

// Returns the elements of the matrix which  returns true for a given function
func Where(matrix [][]float64, function func(x float64) bool) (result []float64) {
	for ii := 0; ii < len(matrix); ii++ {
		for jj := 0; jj < len(matrix[0]); jj++ {
			if function(matrix[ii][jj]) {
				fmt.Println(ii, jj, matrix[ii][jj])
				result = append(result, matrix[ii][jj])
			}
		}
	}
	return
}

// Returns the Congugate Matrix
func ConjugateMatrix(matrix [][]complex128) (result [][]complex128) {
	function := func(number complex128) (result complex128) {
		return cmplx.Conj(number)
	}
	result = ComplexApply(matrix, function)
	return
}
