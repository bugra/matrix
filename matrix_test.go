package matrix

import (
	"testing"
)

const SMALL_FLOAT float64 = 0.0000001

// TODO: Var, std, correlate, corrcoef, covariance
// http://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html

func TestDotProduct(t *testing.T) {
	// 2x3
	firstMatrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	}

	// 3x2
	secondMatrix := [][]float64{
		[]float64{1, 7},
		[]float64{2, 8},
		[]float64{3, 9},
	}

	// 2x2
	actual := [][]float64{
		[]float64{14, 50},
		[]float64{32, 122},
	}

	computed := DotProduct(firstMatrix, secondMatrix)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed[i]); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestMultiplyByScalar(t *testing.T) {
	// 2x3
	firstMatrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	}

	actual := [][]float64{
		[]float64{-1, -2, -3},
		[]float64{-4, -5, -6},
	}

	computed := MultiplyByScalar(firstMatrix, -1)
	for ii := 0; ii < len(firstMatrix); ii++ {
		for jj := 0; jj < len(firstMatrix[ii]); jj++ {
			if actual[ii][jj] != computed[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}
}

func TestAdd(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	secondMatrix := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	actual := [][]float64{
		[]float64{5, 5, 5},
		[]float64{10, 9, 14},
	}

	computed := Add(firstMatrix, secondMatrix)

	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}
}

func TestMultTrans(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	secondMatrix := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	actual := DotProduct(firstMatrix, Transpose(secondMatrix))
	computed := MultTranspose(firstMatrix, secondMatrix)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestMultiply(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	}
	secondMatrix := [][]float64{
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	}

	actual := [][]float64{
		[]float64{4, 10, 18},
		[]float64{28, 40, 54},
	}

	computed := Multiply(firstMatrix, secondMatrix)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestSubtract(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{7, 8, 9},
	}
	secondMatrix := [][]float64{
		[]float64{4, 5, 6},
		[]float64{4, 5, 6},
	}

	actual := [][]float64{
		[]float64{-3, -3, -3},
		[]float64{3, 3, 3},
	}

	computed := Subtract(firstMatrix, secondMatrix)

	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}
}

func TestMatrixTrans(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}

	actual := [][]float64{
		[]float64{3, 9},
		[]float64{2, 5},
		[]float64{1, 7},
	}

	computed := Transpose(firstMatrix)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed[0]); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestSum(t *testing.T) {
	matrix := [][]float64{
		[]float64{0, 1},
		[]float64{0, 5},
	}
	rowSum, ok := Sum(matrix, 0)
	actual := []float64{0, 6}
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(rowSum); ii++ {
		if rowSum[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], rowSum[ii])
		}
	}

	colSum, ok := Sum(matrix, 1)
	if !ok {
		t.Errorf("%b", ok)
	}
	actual = []float64{1, 5}
	for ii := 0; ii < len(colSum); ii++ {
		if colSum[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], colSum[ii])
		}
	}
}

func TestMean(t *testing.T) {
	matrix := [][]float64{
		[]float64{1, 2},
		[]float64{3, 4},
	}
	computed, ok := Mean(matrix, 0)
	if !ok {
		t.Errorf("%b", ok)
	}
	actual := []float64{2, 3}
	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
	computed, ok = Mean(matrix, 1)
	if !ok {
		t.Errorf("%b", ok)
	}
	actual = []float64{1.5, 3.5}
	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
}

func TestMedian(t *testing.T) {
	matrix := [][]float64{
		[]float64{10, 7, 4},
		[]float64{3, 2, 1},
	}

	actual := []float64{6.5, 4.5, 2.5}
	computed, ok := Median(matrix, 0)
	if !ok {
		t.Errorf("%b", ok)
	}

	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
	actual = []float64{7, 2}
	computed, ok = Median(matrix, 1)

	if !ok {
		t.Errorf("%b", ok)
	}

	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
}

func TestMax(t *testing.T) {
	matrix := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 5, 2},
	}

	actual := []float64{2, 5, 4}
	computed, ok := Max(matrix, 0)

	if !ok {
		t.Errorf("%b", ok)
	}

	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}

	actual = []float64{4, 5}
	computed, ok = Max(matrix, 1)

	if !ok {
		t.Errorf("%b", ok)
	}

	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}

}

func TestMin(t *testing.T) {
	matrix := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 5, 2},
	}

	actual := []float64{1, 3, 2}
	computed, ok := Min(matrix, 0)

	if !ok {
		t.Errorf("%b", ok)
	}
	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}

	actual = []float64{2, 1}
	computed, ok = Min(matrix, 1)

	if !ok {
		t.Errorf("%b", ok)
	}

	for ii, _ := range computed {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}

}

func TestSumAll(t *testing.T) {
	m := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	}

	actual := 21.
	computed := SumAll(m)

	if computed != actual {
		t.Errorf("Actual value: %f, computed value: %f", actual, computed)
	}
}

func TestCumulativeSum(t *testing.T) {
	matrix := [][]float64{
		[]float64{2, 1},
		[]float64{2, 5},
	}

	computed, ok := CumulativeSum(matrix, 0)
	actual := []float64{4, 10}
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
	computed, ok = CumulativeSum(matrix, 1)
	actual = []float64{3, 10}
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
}

func TestApply(t *testing.T) {
	m := [][]float64{
		[]float64{4, 2, 1},
		[]float64{8, 3, 6},
	}

	actual := [][]float64{
		[]float64{2, 1, 0.5},
		[]float64{4, 1.5, 3},
	}

	function := func(x float64) float64 {
		return x / 2
	}

	computed := Apply(m, function)

	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed[0]); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Error("Expected computed on pos:", ii, jj, ":", actual[ii][jj], "but obtained:", computed[ii][jj])
			}
		}
	}
}

func TestWhere(t *testing.T) {
	m := [][]float64{
		[]float64{4, 2, 1},
		[]float64{8, 3, 6},
	}
	actual := []float64{4, 8, 6}
	function := func(x float64) bool {
		return x > 3
	}

	computed := Where(m, function)
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Error("Expected computed on pos:", ii, ":", actual[ii], "but obtained:", computed[ii])
		}
	}
}

func TestDeterminant(t *testing.T) {
	matrix := [][]float64{
		[]float64{-2, 2, -3},
		[]float64{-1, 1, 3},
		[]float64{2, 0, -1},
	}

	determinant := Determinant(matrix)
	actual := 18.
	if determinant != actual {
		t.Errorf("Computed Determinant: %f\nActual Determinant: %f", determinant, actual)
	}

	matrix = [][]float64{
		[]float64{1, 3},
		[]float64{4, 2},
	}

	determinant = Determinant(matrix)
	actual = -10.

	if determinant != actual {
		t.Errorf("Computed Determinant: %f\nActual Determinant: %f", determinant, actual)
	}

	matrix = [][]float64{
		[]float64{1, 2, 3},
		[]float64{2, 4, 6},
		[]float64{5, 8, 12},
	}
	actual = 0.
	determinant = Determinant(matrix)
	if determinant != actual {
		t.Errorf("Computed Determinant: %f\nActual Determinant: %f", determinant, actual)
	}
}

func TestMinorMatrix(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	actual := [][]float64{
		[]float64{-2, 0, 1},
		[]float64{9, 2, -3},
		[]float64{5, 1, -2},
	}

	computed := MinorMatrix(m)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed[0]); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestCofactorMatrix(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, -1, -2},
		[]float64{2, 3, 4},
	}

	actual := [][]float64{
		[]float64{1, -3, 1},
		[]float64{-1, -1, 2},
		[]float64{2, -3, 4},
	}

	computed := CofactorMatrix(m)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed[0]); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestInverse(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	actual := [][]float64{
		[]float64{2, 9, -5},
		[]float64{0, -2, 1},
		[]float64{-1, -3, 2},
	}

	computed := Inverse(m)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed[0]); j++ {
			if computed[i][j] != actual[i][j] {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestColumn(t *testing.T) {
	matrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{2, 1, 3},
	}

	computed := column(matrix, 0)

	actual := []float64{1, 3, 2}

	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}

	computed = column(matrix, 1)

	actual = []float64{2, 2, 1}

	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}

	computed = column(matrix, 2)

	actual = []float64{3, 1, 3}

	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			//t.Errorf("Actual value: %f, computed value: %f", actual[ii], computed[ii])
		}
	}
}

func TestDiv(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{2, 1, 3},
	}

	secondMatrix := [][]float64{
		[]float64{4, 5, 6},
		[]float64{6, 5, 4},
		[]float64{4, 6, 5},
	}

	actual := [][]float64{
		[]float64{7.0 / 10.0, 3.0 / 10.0, 0},
		[]float64{-3.0 / 10.0, 7.0 / 10.0, 0},
		[]float64{6.0 / 5.0, 1.0 / 5.0, -1},
	}

	computed := Divide(firstMatrix, secondMatrix)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j]-actual[i][j] > SMALL_FLOAT {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
}

func TestDiagonal(t *testing.T) {
	matrix := [][]float64{
		[]float64{0, 1, 2},
		[]float64{3, 4, 5},
		[]float64{6, 7, 8},
	}
	computed, ok := Diagonal(matrix)
	actual := []float64{0, 4, 8}
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f, in the position: [%d]", actual[ii], computed[ii], ii)
		}
	}
	matrix = [][]float64{
		[]float64{1, 4, 7},
		[]float64{2, 4, 8},
	}
	actual = []float64{1, 4}
	computed, ok = Diagonal(matrix)
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f, in the position: [%d]", actual[ii], computed[ii], ii)
		}
	}
	matrix = [][]float64{
		[]float64{1, 2},
		[]float64{4, 4},
		[]float64{7, 8},
	}
	computed, ok = Diagonal(matrix)
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f, in the position: [%d]", actual[ii], computed[ii], ii)
		}
	}
	matrix = [][]float64{
		[]float64{7},
	}
	actual = []float64{7}
	computed, ok = Diagonal(matrix)
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f, in the position: [%d]", actual[ii], computed[ii], ii)
		}
	}
}

func TestZeros(t *testing.T) {
	dims := []int{3}
	computed, ok := Zeros(dims)
	actual := [][]float64{
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
	}
	if !ok {
		t.Errorf("%b", ok)
	}
	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j]-actual[i][j] > SMALL_FLOAT {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
	dims = []int{2, 4}
	computed, ok = Zeros(dims)
	actual = [][]float64{
		[]float64{0, 0, 0, 0},
		[]float64{0, 0, 0, 0},
	}
	if !ok {
		t.Errorf("%b", ok)
	}
	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j]-actual[i][j] > SMALL_FLOAT {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}

}

func TestEye(t *testing.T) {
	computed, ok := Eye(3)
	actual := [][]float64{
		[]float64{1, 0, 0},
		[]float64{0, 1, 0},
		[]float64{0, 0, 1},
	}
	if !ok {
		t.Errorf("%b", ok)
	}
	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j]-actual[i][j] > SMALL_FLOAT {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}
	computed, ok = Eye(2, 4)
	actual = [][]float64{
		[]float64{1, 0, 0, 0},
		[]float64{0, 1, 0, 0},
	}
	if !ok {
		t.Errorf("%b", ok)
	}
	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j]-actual[i][j] > SMALL_FLOAT {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}

}

func TestConcatenate(t *testing.T) {
	firstMatrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	}

	secondMatrix := [][]float64{
		[]float64{4, 5},
		[]float64{7, 8},
		[]float64{10, 11},
	}

	thirdMatrix := [][]float64{
		[]float64{10, 11, 12},
		[]float64{13, 14, 15},
		[]float64{16, 17, 18},
	}

	actual := [][]float64{
		[]float64{1, 2, 3, 4, 5},
		[]float64{4, 5, 6, 7, 8},
		[]float64{7, 8, 9, 10, 11},
	}

	computed := Concatenate(firstMatrix, secondMatrix, 1)

	for i := 0; i < len(computed); i++ {
		for j := 0; j < len(computed); j++ {
			if computed[i][j]-actual[i][j] > SMALL_FLOAT {
				t.Error("Expected computed on pos:", i, j, ":", actual[i][j], "but obtained:", computed[i][j])
			}
		}
	}

	computed = Concatenate(firstMatrix, thirdMatrix, 0)
	actual = [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
		[]float64{10, 11, 12},
		[]float64{13, 14, 15},
		[]float64{16, 17, 18},
	}
	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed[ii]); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}
}

func TestLowerTriangle(t *testing.T) {
	matrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
		[]float64{10, 11, 12},
	}
	computed := LowerTriangle(matrix)
	actual := [][]float64{
		[]float64{1, 0, 0},
		[]float64{4, 5, 0},
		[]float64{7, 8, 9},
		[]float64{10, 11, 12},
	}
	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed[ii]); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}
}

func TestUpperTriangle(t *testing.T) {
	matrix := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
		[]float64{10, 11, 12},
	}
	computed := UpperTriangle(matrix)
	actual := [][]float64{
		[]float64{1, 2, 3},
		[]float64{0, 5, 6},
		[]float64{0, 0, 9},
		[]float64{0, 0, 0},
	}
	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed[ii]); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}
}

func TestTake(t *testing.T) {
	matrix := [][]float64{
		[]float64{4, 3},
		[]float64{5, 7},
		[]float64{6, 8},
	}
	indices := []int{0, 1, 4}
	computed, ok := Take(matrix, indices)
	actual := []float64{4, 3, 6}
	if !ok {
		t.Errorf("%b", ok)
	}
	for ii := 0; ii < len(computed); ii++ {
		if computed[ii] != actual[ii] {
			t.Errorf("Actual value: %f, computed value: %f, in the position: [%d]", actual[ii], computed[ii], ii)
		}
	}
}

func TestConjugateMatrix(t *testing.T) {
	complexMatrix := [][]complex128{
		[]complex128{1 + 2i},
	}
	computed := ConjugateMatrix(complexMatrix)
	actual := [][]complex128{
		[]complex128{1 - 2i},
	}
	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}

	complexMatrix = [][]complex128{
		[]complex128{3 + 4i, 5 + 7i},
		[]complex128{2 + 3i, 4 + 9i},
	}
	computed = ConjugateMatrix(complexMatrix)
	actual = [][]complex128{
		[]complex128{3 - 4i, 5 - 7i},
		[]complex128{2 - 3i, 4 - 9i},
	}
	for ii := 0; ii < len(computed); ii++ {
		for jj := 0; jj < len(computed); jj++ {
			if computed[ii][jj] != actual[ii][jj] {
				t.Errorf("Actual value: %f, computed value: %f, in the position: [%d][%d]", actual[ii][jj], computed[ii][jj], ii, jj)
			}
		}
	}

}
