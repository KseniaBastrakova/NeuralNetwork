#pragma once
#include <cmath>
class SigmoidFunction
{
public:
	static double Compute( double x )
	{
		double r = ( 1.0 / ( 1.0 + std::exp( -0.5 * x ) ) );
		return r;
	}

   static double ComputeFirstDerivative( double x )
	{
		return 0.5 * Compute( x ) * ( 1.0 - Compute( x ) );
	}
};