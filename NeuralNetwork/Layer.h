#pragma once
#include "Neuron.h"
#include <vector>
#include <fstream>

class Layer
{
public:
	Layer( int theInputDemention, std::vector<std::vector<double>> weights ):
		inputDimension( theInputDemention ){
		for ( int i = 0; i < inputDimension; i++ )
			Neurons.push_back( new Neuron( weights[i]) );
	}
	std::vector<double> LastOutput;
	std::vector<Neuron*> Neurons;
	int inputDimension;

public:
	void Compute( std::vector<double> inputVector )
	{
		LastOutput.clear();
		for ( int i = 0; i < Neurons.size(); i++ )
		{		
			LastOutput.push_back( Neurons[i]->Activate( inputVector ) );
		}
	}
	std::vector<double> GetLastOutput()
	{
		return LastOutput;
	}
	void SetLastOutput( std::vector<double> theLastoutput )
	{
		LastOutput = theLastoutput;
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

	size_t GetNeuronSize()
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
			Neurons.push_back( new Neuron( weightsInput ) );
		}
	
	}
};