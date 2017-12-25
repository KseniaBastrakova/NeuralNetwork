#include "ActivateFunction.h"
#include <cmath>


namespace {
class SigmoidFunction
{
public:
	static double Compute( double x, const std::vector<double>& neyrons )
	{
		double r = ( 1.0 / ( 1.0 + std::exp( -x ) ) );
		return r;
	}

	static double ComputeFirstDerivative( double x, const std::vector<double>& neyrons )
	{
		return Compute( x, neyrons ) * ( 1.0 - Compute( x, neyrons ) );
	}
};

class SoftMax
{
public:
	static double Compute( double x, const std::vector<double>& neyrons )
	{
		double numerator = exp( x );
		double denominator = 0;
		for ( int i = 0; i < neyrons.size(); i++ )
		{
			denominator += exp( neyrons[i] );
		}
		return numerator / denominator;
	}

	static double ComputeFirstDerivative( double x, const std::vector<double>& neyrons )
	{
		double y = Compute( x, neyrons );
		return y* ( 1 - y );
	}
};
}

ActivationFunction::ActivationFunction( ComputeFunction theValue, ComputeFirstDerivativeFunction
theDerivative):
	value(theValue), derivative(theDerivative){}

double ActivationFunction::Compute( double x, const std::vector<double>& neyrons )
{
	return ( *value )( x, neyrons );
}

double ActivationFunction::ComputeFirstDerivative( double x, const std::vector<double>& neyrons )
{
	return ( *derivative )( x, neyrons );
}

ActivationFunction ActivationFunction::Sigmoid()
{
	return ActivationFunction( &SigmoidFunction::Compute, &SigmoidFunction::ComputeFirstDerivative );
}

ActivationFunction ActivationFunction::SoftMax()
{
	return ActivationFunction( &SoftMax::Compute, &SoftMax::ComputeFirstDerivative );
}


