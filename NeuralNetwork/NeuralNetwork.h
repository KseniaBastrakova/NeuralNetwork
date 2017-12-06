#pragma once

#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "Layer.h"

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
		std::vector<std::vector<double>> weightFurst( size[0] );
		layers.push_back( new Layer( size[0], weightFurst ));
		for ( size_t k = 1; k < numLayers; k++ )
		{
			std::vector<std::vector<double>> weightsN;
			weightsN.resize( size[k] );
			for ( size_t i = 0; i < weightsN.size(); i++ )
				weightsN[i].resize( size[k - 1] );

			for ( int i = 0; i < weightsN.size(); i++ )
				for ( int j = 0; j < weightsN[i].size(); j++ )
					weightsN[i][j] = ( rand() % 100 - 50 ) / 100.0;
				layers.push_back( new Layer( size[k], weightsN ) );
		}
	}

	std::vector<double> Perform( std::vector<double> inputVaues )
	{
		layers[0]->SetLastOutput( inputVaues );
		for ( int i = 1; i < layers.size(); i++ )
			layers[i]->Compute( layers[i - 1]->GetLastOutput() );
		return layers[layers.size() - 1]->GetLastOutput();
	}

	double CountError( std::vector<std::vector<double>> expectedValues, std::vector<std::vector<double>>
					   currentValue )
	{
		double error = 0.;
		for ( int i = 0; i < currentValue.size(); i++ )
			for ( int j = 0; j < currentValue[i].size(); j++ )
			{
				error += fabs( currentValue[i][j] - expectedValues[i][j] );
			}
		return error;
	}
	double TestTrain( std::vector<std::vector<double>> inputValues, std::vector<std::vector<double>> expectedValues )
	{
		std::vector<std::vector<double>> results;
		for ( size_t i = 0; i < inputValues.size(); i++ )
		{
			results.push_back( Perform( inputValues[i] ) );
		}
		for ( size_t i = 0; i < results.size(); i++ )
		{
			int maxIdx = std::distance( results[i].begin(), std::max_element( results[i].begin(), results[i].end() ) );
			for ( size_t j = 0; j < results[i].size(); j++ )
				results[i][j] = 0;
			results[i][maxIdx] = 1;
		}
		double error = CountError( results, expectedValues );
		return error/2.;
	}

	Layer* GetLayer( int idx )
	{
		return layers[idx];
    }

	size_t GetLayerSize()
	{
		return numLayers;
	}

	std::vector<Layer*> GetLayers()
	{
		return layers;
	}

	
	std::vector<Layer*> layers;
};