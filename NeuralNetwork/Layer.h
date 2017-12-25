#pragma once
#include "Neuron.h"
#include <vector>
#include <fstream>
#include "LayerData.h"
#include "ActivateFunction.h"

class Layer
{
public:
	Layer( int theInputDemention, std::vector<std::vector<double>> weights, std::vector<double> biases,
		   ActivationFunction function ):
		inputDimension( theInputDemention ){
		for ( int i = 0; i < inputDimension; i++ )
			Neurons.push_back( new Neuron( weights[i], biases[i], function) );
	}
	std::vector<Neuron*> Neurons;
	int inputDimension;

public:
	LayerData Compute( const LayerData& previousLayer )
	{
		LayerData result;
		for ( size_t i = 0; i < Neurons.size(); i++ )
			result.nets.push_back( Neurons[i]->NET( previousLayer ) );
		for ( size_t i = 0; i < Neurons.size(); i++ )
			result.valuesAfterActivation.push_back( Neurons[i]->Activate( result.nets[i], result.nets ) );
		return result;
	}
	void SetWeights( std::vector<std::vector<double>> theWeightsNeuron )
	{
		for ( int i = 0; i < inputDimension; i++ )
			Neurons[i]->SetWeights( theWeightsNeuron[i] );
	}

	std::vector<double> GetWeightsNeuron( int idx )
	{
		return Neurons[idx]->GetWeights();
	}

	void SetWeightsNeuron( int idx, std::vector<double> newWeights )
	{
		return Neurons[idx]->SetWeights( newWeights );
	}

	void SetBias( int idx, double theBias )
	{
		Neurons[idx]->SetBias( theBias );
	}

	std::vector<Neuron*> GetNeurons() const
	{
		return Neurons;
	}

	size_t GetNeuronSize() const
	{
		return Neurons.size();
	}

	void LayerSerializer( fstream &out )
	{
		out << Neurons.size() << " ";
		for ( size_t i = 0; i < Neurons.size(); i++ )
			Neurons[i]->WeightsSerializer( out );
		out << std::endl;
	}

	void LayerReader( fstream &out, std::vector<size_t> sizeOfLayers )
	{
		int sizeNeurons = 0;
		out >> sizeNeurons;
		for ( int j = 0; j < sizeNeurons; j++ )
		{
			std::vector<double> weightsInput;
			int numberWeights = 0;
			out << numberWeights;
			for ( int i = 0; i < numberWeights; i++ )
			{
				double value = 0.;
				out >> value;
				weightsInput.push_back( value );
			}
//			Neurons.push_back( new Neuron( weightsInput ) );
		}
	
	}
};