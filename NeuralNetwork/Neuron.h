#ifndef NEIRON
#define  NEIRON

#include <vector>
#include <fstream>
#include "ActivateFunction.h"
using namespace std;

class Neuron
{
	std::vector<double> weights;
	double Bias;

	double LastNET;
public:
	Neuron( std::vector<double> theWeights): Bias( 0. ), weights( theWeights ){}
	double Activate( std::vector<double> inputVector )
	{
		SigmoidFunction func;
		return func.Compute( NET( inputVector ) );
	}
	double NET( std::vector<double> inputVector )
	{
		double result = 0.0;
		for ( int i = 0; i < inputVector.size(); i++ )
			result += inputVector[i] * weights[i];
		result += Bias;
		LastNET = result;
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

	void SetLastNET( double theLastNET )
	{
		LastNET = theLastNET;
	}

	double GetLastNET()
	{
		return LastNET;
	}

	void WeightsSerializer( fstream& out )
	{
		out << weights.size() << " ";
		for ( int i = 0; i < weights.size(); i++ )
			out << weights[i]<<"  ";
	}

};
#endif