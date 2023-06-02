package bcStat

import (
	"math"
)

type DataPoint struct {
	X float64
	Y float64
}

// SimpleLinearRegression calculates the simple linear regression for the given trendline.
// It computes the slope (A), intercept (B), and R-squared (r2) values using the LinearRegression function.
//
// Parameters:
//   - trendline: A slice of DataPoint structs representing the trendline data points.
//
// Returns:
//   - A: The slope of the regression line.
//   - B: The intercept of the regression line.
//   - r2: The R-squared value, indicating the goodness of fit of the regression line.
//
// The SimpleLinearRegression function internally calls the LinearRegression function with default values for maxOutliers and forceIntercept.
func SimpleLinearRegression(trendline []DataPoint) (float64, float64, float64) {
	A, B, r2, _ := LinearRegression(trendline, 0, math.NaN())
	return A, B, r2
}

// LinearRegression calculates the linear regression for the given trendline.
// It supports removing outliers up to the specified maximum outliers and allows forcing a specific intercept value.
// It returns the slope (A), intercept (B), R-squared (r2), and the indexes of removed outliers.
//
// Parameters:
//   - trendline: A slice of DataPoint structs representing the trendline data points.
//   - maxOutliers: The maximum number of outliers to remove during the regression. Set to 0 to disable outlier removal.
//   - forceIntercept: An optional parameter to force a specific intercept value. Pass math.NaN() to perform a regular regression.
//
// Returns:
//   - A: The slope of the regression line.
//   - B: The intercept of the regression line.
//   - r2: The R-squared value, indicating the goodness of fit of the regression line.
//   - removedIndexes: A slice of integers representing the indexes of the removed outliers from the original trendline.
//     If no outliers were removed, this will be an empty slice ([]int{}).
func LinearRegression(trendline []DataPoint, maxOutliers int, forceIntercept float64) (float64, float64, float64, []int) {
	if len(trendline) < 2 {
		return math.NaN(), math.NaN(), math.NaN(), nil
	}

	var removedIndexes []int

	A, B := doRegression(trendline, forceIntercept)

	// Calculate R2 value
	r2 := calculateR2(trendline, forceIntercept)

	if r2 < 0.95 && maxOutliers > 0 {
		maxOutliers = maxOutliers - 1
		removedIndex := removeOutlier(trendline, forceIntercept)
		if removedIndex != -1 {
			trendline = append(trendline[:removedIndex], trendline[removedIndex+1:]...)
			A, B, r2, removedIndexes = LinearRegression(trendline, maxOutliers, forceIntercept)
		}
		removedIndexes = append(removedIndexes, removedIndex)
	}

	return A, B, r2, removedIndexes
}

func doRegression(trendline []DataPoint, forceIntercept float64) (float64, float64) {
	// Calculate sum of X, sum of Y, sum of XY, sum of X^2
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumXSquare := 0.0
	n := float64(len(trendline))

	for _, point := range trendline {
		sumX += point.X
		sumY += point.Y
		sumXY += point.X * point.Y
		sumXSquare += point.X * point.X
	}

	// Calculate slope (A) and intercept (B)
	var A, B float64
	if math.IsNaN(forceIntercept) {
		A = (n*sumXY - sumX*sumY) / (n*sumXSquare - sumX*sumX)
		B = (sumY - A*sumX) / n
	} else {
		A = (sumXY - forceIntercept*sumX) / sumXSquare
		B = forceIntercept
	}
	return A, B
}

func removeOutlier(trendline []DataPoint, forceIntercept float64) int {
	highestR2 := calculateR2(trendline, forceIntercept)
	highestR2Index := -1

	//debug: fmt.Printf("With nothing removed, R2 = %.4f\n", highestR2)

	for i := range trendline {
		temp := make([]DataPoint, len(trendline))
		copy(temp, trendline)
		temp = append(temp[:i], temp[i+1:]...)

		r2 := calculateR2(temp, forceIntercept)

		//debug: fmt.Printf("With %.2f removed, R2 = %.4f\n", trendline[i], r2)

		if r2 > highestR2 {
			highestR2 = r2
			highestR2Index = i
		}
	}

	//debug: fmt.Printf("Removed X:%.2f, R2 is now %.4f\n", trendline[highestR2Index].X, highestR2)

	return highestR2Index
}

func calculateR2(trendline []DataPoint, forceIntercept float64) float64 {
	if len(trendline) < 2 {
		return math.NaN()
	}

	A, B := doRegression(trendline, forceIntercept)

	// Calculate sum of Y, sum of Y^2, sum of predictedY, and sum of predictedY^2
	sumY := 0.0
	sumYSquared := 0.0
	sumPredictedY := 0.0
	sumPredictedYSquared := 0.0
	n := float64(len(trendline))

	for _, point := range trendline {
		sumY += point.Y
		sumYSquared += point.Y * point.Y
		predictedY := A*point.X + B
		sumPredictedY += predictedY
		sumPredictedYSquared += predictedY * predictedY
	}

	// Calculate predicted Y and sum of predictedY^2
	sumYDiffSquared := 0.0
	for _, point := range trendline {
		predictedY := A*point.X + B
		diff := point.Y - predictedY
		sumYDiffSquared += diff * diff
	}

	// Calculate R2 value
	r2 := 1 - (sumYDiffSquared / (sumYSquared - (sumY*sumY)/n))

	return r2
}
