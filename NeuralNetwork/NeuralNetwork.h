#pragma once

#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "Layer.h"
#include "NetworkData.h"
#include "LayerData.h"
#include "ActivateFunction.h"

class NeuralNetwork
{
	std::vector<size_t> size;
	size_t numLayers;
public:
	NeuralNetwork( std::vector<size_t> theSize, std::vector<Layer*> theLayers ):
		size( theSize ),
		numLayers( theSize.size() ),
		layers( theLayers )
	{
	}

	NeuralNetwork( std::vector<size_t> theSize ):
		size( theSize ),
		numLayers( theSize.size() )
	{
		std::vector<std::vector<double>> weightFirst( size[0] );
		std::vector<double> biasesFirst( size[0] );
		layers.push_back( new Layer( size[0], weightFirst, biasesFirst, ActivationFunction::Sigmoid()));
		for ( size_t k = 1; k < numLayers; k++ )
		{
			std::vector<std::vector<double>> weightsN;
			weightsN.resize( size[k] );
			for ( size_t i = 0; i < weightsN.size(); i++ )
				weightsN[i].resize( size[k - 1] );

			for ( size_t i = 0; i < weightsN.size(); i++ )
				for ( size_t j = 0; j < weightsN[i].size(); j++ )
					weightsN[i][j] = ( rand() % 100 - 50 ) / 100.0;
			
			std::vector<double> biases( size[k] );
			for ( size_t i = 0; i < biases.size(); i++ )
				biases[i] = ( rand() % 100 - 50 ) / 100.0;

			if ( k < numLayers - 1 )
				layers.push_back( new Layer( size[k], weightsN, biases, ActivationFunction::Sigmoid() ) );
			else
				layers.push_back( new Layer( size[k], weightsN, biases, ActivationFunction::SoftMax() ));
		}
	}

	NetworkData Perform( std::vector<double> inputValues ) const
	{
		LayerData inputLayerData;
		inputLayerData.nets = inputValues;
		inputLayerData.valuesAfterActivation = inputValues;
		NetworkData result;
		result.add( inputLayerData );

		for ( size_t i = 1; i < layers.size(); i++ )
		{
			LayerData layerData = layers[i]->Compute( result[i - 1] );
			result.add( layerData );
		}
		return result;
	}

	//double CountError( std::vector<std::vector<double>> expectedValues, std::vector<std::vector<double>>
	//				   currentValue )
	//{
	//	double error = 0.;
	//	for ( int i = 0; i < currentValue.size(); i++ )
	//		for ( int j = 0; j < currentValue[i].size(); j++ )
	//		{
	//			error += fabs( currentValue[i][j] - expectedValues[i][j] );
	//		}
	//	return error;
	//}

	Layer* GetLayer( int idx )
	{
		return layers[idx];
    }

	size_t GetLayerSize() const
	{
		return numLayers;
	}

	std::vector<Layer*> GetLayers()
	{
		return layers;
	}

	
	std::vector<Layer*> layers;
};