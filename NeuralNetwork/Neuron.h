#ifndef NEIRON
#define  NEIRON

#include <vector>
#include <fstream>
#include "ActivateFunction.h"
#include "LayerData.h"
using namespace std;

class Neuron
{
	std::vector<double> weights;
	double Bias;
	ActivationFunction function;
public:
	Neuron( std::vector<double> theWeights, double theBias, ActivationFunction theFunction): 
		Bias( theBias ), weights( theWeights ), function(theFunction){}
	double Activate( double net, std::vector<double> nets )
	{
		return function.Compute( net, nets );
	}
	ActivationFunction GetActivationFunction()
	{
		return function;
	}
	double NET( const LayerData& previousLayer )
	{
		double result = 0.0;
		for ( size_t i = 0; i < previousLayer.valuesAfterActivation.size(); i++ )
			result += previousLayer.valuesAfterActivation[i] * weights[i];
		result += Bias;
		return result;
	}

	std::vector<double> GetWeights()
	{
		return weights;
	}
	void SetWeights( std::vector<double> theWeights )
	{
		weights = theWeights;
	}


	void SetBias( double theBias )
	{
		Bias = theBias;
	}

	double GetBias()
	{
		return Bias;
	}

	void WeightsSerializer( fstream& out )
	{
		out << weights.size() << " ";
		for ( size_t i = 0; i < weights.size(); i++ )
			out << weights[i]<<"  ";
	}

};
#endif