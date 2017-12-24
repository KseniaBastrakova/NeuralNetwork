#include "ErrorFunction.h"
#include "NeuralNetwork.h"
#include "ActivateFunction.h"
#include "LayerData.h"
#include "NetworkData.h"
#include "LossFunction.h"
#include <limits>
#include <iostream>
#include <vector>



class LearningAlgorithm
{
public:
	LearningAlgorithm( NeuralNetwork* theNetWork, std::vector<double> theStartedValues, std::vector<double> theExpectedValues,
					   double theGradientSpeedParam ):
		net(theNetWork), inputVector(theStartedValues), expectedValues(theExpectedValues), gradientSpeedParam(theGradientSpeedParam){}

	void Learn()
	{
		NetworkData output = net->Perform( inputVector );
		/// DEBUG
		//output[2].valuesAfterActivation[0] = 0.2698;
		//output[2].valuesAfterActivation[1] = 0.3223;
		//output[2].valuesAfterActivation[2] = 0.4078;

			/// END DEBUG
		UpdateLastLayer( output );
		for ( int i = net->GetLayerSize() - 2; i >= 1; i-- )
		{
			UpdateLayer( i, output );
		}
	}
	void UpdateLayer( size_t idx, const NetworkData& data )
	{
		const LayerData& layerData = data[idx];
		const LayerData& prevLayerData = data[idx - 1];
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
		for ( size_t i = 0; i < numNeurons; i++ )
		{
			Neuron* neuron = layer->Neurons[i];
			ActivationFunction function = neuron->GetActivationFunction();
			double currentSigma = 0.;
			for ( size_t j = 0; j < prevSigma.size(); j++ )
			{
				currentSigma += prevSigma[j] * prevWeights[j][i];

			}
			sigma[i] = currentSigma * function.ComputeFirstDerivative( layerData.nets[i], layerData.nets );
			weights[i] = neuron->GetWeights();
			Layer* prevLayer = net->layers[idx - 1];
			std::vector<double> newWeights = weights[i];
			for ( size_t j = 0; j < newWeights.size(); j++ )
			{
				newWeights[j] += gradientSpeedParam * sigma[i] * prevLayerData.valuesAfterActivation[j];
			}
			neuron->SetWeights( newWeights );
			bias[i] = neuron->GetBias();
			neuron->SetBias( neuron->GetBias() + gradientSpeedParam * sigma[i] );
		}
	}
	void UpdateLastLayer( const NetworkData& data )
	{
		Layer* layer = net->layers.back();
		const LayerData& layerData = data[net->layers.size() - 1];
		const LayerData& prevLayerData = data[net->layers.size() - 2];
		sigma = std::vector<double>( expectedValues.size(), 0.0 );
		weights.resize( expectedValues.size() );
		bias.resize( expectedValues.size() );

		for ( size_t i = 0; i < expectedValues.size(); i++ )
		{
			Neuron* neuron = layer->Neurons[i];
			ActivationFunction function = neuron->GetActivationFunction();
			sigma[i] = ( expectedValues[i] - layerData.valuesAfterActivation[i] )  /**
				function.ComputeFirstDerivative( layerData.nets[i], layerData.nets ) */;
			weights[i] = neuron->GetWeights();
			std::vector<double> newWeights = weights[i];
			for ( size_t j = 0; j < newWeights.size(); j++ )
			{
				newWeights[j] += gradientSpeedParam * sigma[i] * prevLayerData.valuesAfterActivation[j];
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
			LearnIteration();
			error = currentError;
		    currentError = GetError( *netWork, inputValues, expectedValues );
			i++;
			std::cout << "Training iteration " << i << ": cross entropy = " << currentError << ", accuracy = "
				<< GetAccuracyPercent(*netWork, inputValues, expectedValues) << "%" << std::endl;
		}
	}
private:
	void LearnIteration()
	{   
		for ( size_t i = 0; i < inputValues.size(); i++ )
		{
			LearningAlgorithm alg( netWork, inputValues[i], expectedValues[i], gradientSpeedParam );
			alg.Learn();
		}
	}
	bool IsOver( int i, double lastError, double currentError)
	{
		return i > 100;// || std::fabs( lastError - currentError ) / lastError < netError;
	}
	std::vector<std::vector<double>> inputValues;
	std::vector<std::vector<double>> expectedValues;
	NeuralNetwork* netWork;
	double	netError;
	double gradientSpeedParam;
};


