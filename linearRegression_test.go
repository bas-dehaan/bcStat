package bcStat

import (
	"math"
	"testing"
)

func round(x float64, digit float64) float64 {
	return math.Round(x*math.Pow(10, digit)) / math.Pow(10, digit)
}

func TestSimpleLinearRegression(t *testing.T) {
	trendline := []DataPoint{
		{X: 1, Y: 100},
		{X: 2, Y: 200},
		{X: 3, Y: 300},
		{X: 4, Y: 400},
		{X: 5, Y: 500},
		{X: 6, Y: 600},
		{X: 7, Y: 700},
		{X: 8, Y: 800},
	}

	A, B, r2 := SimpleLinearRegression(trendline)
	if A != 100 || B != 0 || r2 != 1 {
		t.Errorf("Expected A=100, B=0, r2=1, got A=%.2f, B=%.2f, r2=%.2f", A, B, r2)
	}

	trendline = []DataPoint{
		{X: 1, Y: 0},
		{X: 2, Y: 50},
		{X: 3, Y: 100},
		{X: 4, Y: 200},
		{X: 5, Y: 400},
		{X: 6, Y: 800},
		{X: 7, Y: 1600},
		{X: 8, Y: 3200},
	}

	A, B, r2 = SimpleLinearRegression(trendline)
	if round(A, 2) != 386.31 || round(B, 2) != -944.64 || round(r2, 2) != 0.73 {
		t.Errorf("Expected A=386.31, B=-944.64, r2=0.73, got A=%.2f, B=%.2f, r2=%.2f", A, B, r2)
	}
}

func TestLinearRegression(t *testing.T) {
	trendline := []DataPoint{
		{X: 1, Y: 200},
		{X: 2, Y: 300},
		{X: 3, Y: 0},
		{X: 4, Y: 500},
		{X: 5, Y: 600},
		{X: 6, Y: 0},
		{X: 7, Y: 800},
		{X: 8, Y: 900},
	}

	A, B, r2, out := LinearRegression(trendline, 9, math.NaN())
	if A != 100 || B != 100 || r2 != 1 || len(out) != 2 || out[0] != 2 || out[1] != 5 {
		t.Errorf("Expected A=100, B=100, r2=1, len(out)=2, out[0]=2, out[1]=5; got A=%.2f, B=%.2f, r2=%.2f, len(out)=%d, out[0]=%d, out[1]=%d", A, B, r2, len(out), out[0], out[1])
	}

	trendline = []DataPoint{
		{X: 1, Y: 200},
		{X: 2, Y: 300},
		{X: 3, Y: 0},
		{X: 4, Y: 500},
		{X: 5, Y: 600},
		{X: 6, Y: 0},
		{X: 7, Y: 800},
		{X: 8, Y: 900},
	}

	A, B, r2, out = LinearRegression(trendline, 1, 12)
	if round(A, 2) != 108.57 || round(B, 2) != 12 || round(r2, 2) != 0.79 || len(out) != 1 || out[0] != 5 {
		t.Errorf("Expected A=108.57, B=12, r2=0.79, len(out)=1, out[0]=5; got A=%.2f, B=%.2f, r2=%.2f, len(out)=%d, out[0]=%d", A, B, r2, len(out), out[0])
	}
}
