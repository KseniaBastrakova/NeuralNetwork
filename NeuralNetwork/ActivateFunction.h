#pragma once
#include <vector>

class ActivationFunction
{
public:
	double Compute( double x, const std::vector<double>& neyrons );
	double ComputeFirstDerivative( double x,const std::vector<double>& neyrons );
	static ActivationFunction Sigmoid();
	static ActivationFunction SoftMax();
private:
	typedef double( *ComputeFunction )( double, const std::vector<double>& );
	typedef double( *ComputeFirstDerivativeFunction )( double, const std::vector<double>& );
	ActivationFunction( ComputeFunction, ComputeFirstDerivativeFunction );
	ComputeFunction value;
	ComputeFirstDerivativeFunction derivative;

};
