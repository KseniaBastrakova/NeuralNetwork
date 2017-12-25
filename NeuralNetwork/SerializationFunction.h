#pragma once
#include <vector>
#include <fstream>
#include "Layer.h"
#include "NeuralNetwork.h"

void PrintNetwork( int numLayers, std::vector<size_t> sizeLayers, std::vector<Layer*>layers, std::string pathSaveFile )
{
	fstream out( pathSaveFile );
	out << numLayers<<" ";
	for ( size_t i = 0; i < sizeLayers.size(); i++ )
	{
		out << sizeLayers[i] << " ";
	}
	for ( size_t i = 0; i < layers.size(); i++ )
	{
		layers[i]->LayerSerializer( out );
	}

}

void ReadNetwork( NeuralNetwork* network, std::string pathNetwork )
{
	int numLayers = 0;
	fstream out( pathNetwork );
	out >> numLayers;
	std::vector<size_t> size_Layers;

	for ( int i = 0; i < numLayers; i++ )
	{
		size_t value;
		out >> value;
		size_Layers.push_back( value );
	}
	std::vector<Layer*> layers;
	for ( size_t i = 0; i < size_Layers.size(); i++ )
	{
	//	std::vector<std::vector<double>> values;
	////	Layer *newLayer = new Layer( 0, values );
	//	newLayer->LayerReader( out, size_Layers );
	//	layers.push_back( newLayer );
	}
	network = new NeuralNetwork( size_Layers, layers );
}
