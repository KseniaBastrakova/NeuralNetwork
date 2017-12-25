#pragma once
#include <cmath>
class SigmoidFunction
{
private:
	double _alpha;
public:
	static double Compute( double x )
	{
		double r = ( 1 / ( 1 + std::exp( -1 * 0.5 * x ) ) );
		//return r == 1f ? 0.9999999f : r;
		return r;
	}

   static double ComputeFirstDerivative( double x )
	{
		return 0.5 * Compute( x ) * ( 1 - Compute( x ) );
	}
};