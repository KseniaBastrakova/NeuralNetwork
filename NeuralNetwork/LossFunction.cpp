#include "LossFunction.h"
#include "NeuralNetwork.h"

#include <cmath> 
#include <algorithm>


namespace {
double CrossEnthopy( const std::vector<double>& actual,
					 const std::vector<double>& expected )
{
	double error = 0.;
	for ( size_t i = 0; i < actual.size(); i++ )
	{
		// avoid 0 * infinity = nan
		if (actual[i])
			error += expected[i] * log( actual[i] );
	}
	return -error;
}
}

double GetError( const NeuralNetwork& net, const std::vector<std::vector<double>>& inputValues,
				 const std::vector<std::vector<double>>& expectedResults )
{
	double sumError = 0.;
	for ( size_t i = 0; i < inputValues.size(); i++ )
	{
		NetworkData data = net.Perform( inputValues[i] );
		sumError+= CrossEnthopy( data[net.GetLayerSize() - 1].valuesAfterActivation, expectedResults[i] );
	}
	return sumError / inputValues.size();
}

double GetAccuracyPercent( const NeuralNetwork& net, const std::vector<std::vector<double>>& inputValues,
					const std::vector<std::vector<double>>& expectedResults )
{
	int numCorrect = 0;
	for ( size_t i = 0; i < inputValues.size(); i++ )
	{
		NetworkData data = net.Perform( inputValues[i] );
		std::vector<double>& result = data[net.GetLayerSize() - 1].valuesAfterActivation;
		int resultClass = std::distance( result.begin(), std::max_element( result.begin(), result.end() ) );
		if ( expectedResults[i][resultClass] == 1.0 )
			numCorrect++;
	}
	return 100.0 * (double)numCorrect / inputValues.size();
}