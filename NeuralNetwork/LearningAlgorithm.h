#include "ErrorFunction.h"
#include "NeuralNetwork.h"
#include "ActivateFunction.h"
#include <limits>
#include <iostream>
#include <vector>



class LearningAlgorithm
{
public:
	LearningAlgorithm( NeuralNetwork* theNetWork, std::vector<double> theStartedValues, std::vector<double> theExpectedValues,
					   double theGradientSpeedParam ):
		net(theNetWork), inputVector(theStartedValues), expectedValues(theExpectedValues), gradientSpeedParam(theGradientSpeedParam){}



	std::vector<double> LearnInternalLayer( Layer* currentLayer, std::vector<double> inputErrors)
	{
		std::vector<double> weights;
		std::vector<double> errors;
		size_t size = currentLayer->GetNeuronSize();
		for ( int i = 0; i < size; i++ )
		{
			double error = 0.;
			double neyronInput = 0.;
			for ( int j = 0; j < size; j++ )
			{
				 error += inputErrors[i] * currentLayer->GetWeightsNeuron( i )[j];
				 neyronInput += currentLayer->GetWeightsNeuron( i )[j] * inputVector[i];
			}
			for ( int j=0; j<size; j++ )
			weights.push_back( error * SigmoidFunction::ComputeFirstDerivative( neyronInput ) );
			currentLayer->SetWeightsNeuron(i, weights );
			currentLayer->SetBias( i, error * gradientSpeedParam );
			errors.push_back( error );
		}
		return errors;
	}

	void Learn()
	{
		std::vector<double> output = net->Perform( inputVector );
		UpdateLastLayer();
		for ( int i = net->GetLayerSize() - 2; i >= 1; i-- )
		{
			UpdateLayer( i );
		}
	}
	void UpdateLayer( size_t idx )
	{
		Layer* layer = net->layers[idx];
		size_t numNeurons = layer->GetNeuronSize();
		std::vector<double> prevSigma = sigma;
		std::vector<std::vector<double>> prevWeights = weights;
		std::vector<double> prevBias = bias;

		sigma = std::vector<double>( numNeurons, 0.0 );
		weights.clear();
		weights.resize( numNeurons );
		bias.clear();
		bias.resize( numNeurons );
		for ( int i = 0; i < numNeurons; i++ )
		{
			Neuron* neuron = layer->Neurons[i];
			double currentSigma = 0.;
			for ( int j = 0; j < prevSigma.size(); j++ )
			{
				currentSigma += prevSigma[j] * prevWeights[j][i];

			}
			sigma[i] = currentSigma * SigmoidFunction::ComputeFirstDerivative( neuron->GetLastNET() );
			weights[i] = neuron->GetWeights();
			Layer* prevLayer = net->layers[idx - 1];
			std::vector<double> newWeights = weights[i];
			for ( int j = 0; j < newWeights.size(); j++ )
			{
				newWeights[j] += gradientSpeedParam * sigma[i] * prevLayer->GetLastOutput()[j];
			}
			neuron->SetWeights( newWeights );
			bias[i] = neuron->GetBias();
			neuron->SetBias( neuron->GetBias() + gradientSpeedParam * sigma[i] );
		}
	}
	void UpdateLastLayer()
	{
		Layer* layer = net->layers.back();
		sigma = std::vector<double>( expectedValues.size(), 0.0 );
		weights.resize( expectedValues.size() );
		bias.resize( expectedValues.size() );
		for ( size_t i = 0; i < expectedValues.size(); i++ )
		{
			Neuron* neuron = layer->Neurons[i];
			sigma[i] = ( expectedValues[i] - layer->LastOutput[i] ) *
				SigmoidFunction::ComputeFirstDerivative( neuron->GetLastNET() );
			weights[i] = neuron->GetWeights();
			Layer* prevLayer = net->layers[net->layers.size() - 2];
			std::vector<double> newWeights = weights[i];
			for ( int j = 0; j < newWeights.size(); j++ )
			{
				newWeights[j] += gradientSpeedParam * sigma[i] * prevLayer->GetLastOutput()[j];
			}
			neuron->SetWeights( newWeights );
			bias[i] = neuron->GetBias();
			neuron->SetBias( neuron->GetBias() + gradientSpeedParam * sigma[i] );
		}
	}
private:
	std::vector<double> inputVector;
	std::vector<double> expectedValues;
	std::vector<double> sigma;
	std::vector<std::vector<double>> weights;
	std::vector<double> bias;
	NeuralNetwork* net;
	double gradientSpeedParam;
};

class Training
{
public:
	Training( NeuralNetwork* theNetWork, std::vector<std::vector<double>> theInputValues, 
			  std::vector<std::vector<double>> theExpectedValues, double theNetError, double theGradientSpeedParam ):
		inputValues( theInputValues), expectedValues(theExpectedValues), netWork(theNetWork),netError(theNetError),
		gradientSpeedParam(theGradientSpeedParam){}
	void Perform()
	{
		int i = 0;
		double error = std::numeric_limits<double>::infinity();
		double	currentError = std::numeric_limits<double>::infinity();
		
		while ( !IsOver(i, error, currentError) )
		{
			std::vector<std::vector<double>> result = LearnIteration();
			error = currentError;
		    currentError = CountError( expectedValues, result );
			i++;
		//	std::cout << currentError << "  " << std::endl;;
		}
	}
	double CountError( std::vector<std::vector<double>> expectedValues, std::vector<std::vector<double>>
					   currentValue )
	{
		//double error = 0.;
		//for ( int i = 0; i < currentValue.size(); i++ )
		//	for ( int j = 0; j < currentValue[i].size(); j++ )
		//	{
		//		error += ( currentValue[i][j] - expectedValues[i][j] ) * ( currentValue[i][j] - expectedValues[i][j] );
		//	}
		//return error;

		double sumError = 0.0;
		for ( int i = 0; i<expectedValues.size(); i++ )
		{
			for ( int j = 0; j < expectedValues[i].size(); j++ )
			{
				if ( currentValue[i][j] >0.000001 )
				sumError += log( currentValue[i][j] ) * expectedValues[i][j];

			}
		}
		return -1.0 * sumError / expectedValues.size();
	}
private:
	std::vector<std::vector<double>> LearnIteration()
	{   
		std::vector<std::vector<double>> currentValues;
		for ( int i = 0; i < inputValues.size(); i++ )
		{
			 
			LearningAlgorithm alg( netWork, inputValues[i], expectedValues[i], gradientSpeedParam );
			alg.Learn();
			currentValues.push_back( netWork->Perform( inputValues[i] ) );
		}
		return currentValues;
	}
	bool IsOver( int i, double lastError, double currentError)
	{
		return i > 1000 || std::fabs(lastError - currentError)/lastError < netError;
	}
	std::vector<std::vector<double>> inputValues;
	std::vector<std::vector<double>> expectedValues;
	NeuralNetwork* netWork;
	double	netError;
	double gradientSpeedParam;
};


