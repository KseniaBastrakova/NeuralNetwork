#pragma once
#include <vector>
#include "LayerData.h"
class NetworkData
{
public:
	LayerData& operator[]( int idx )
	{
		return layerData[idx];
	}
	const LayerData& operator[]( int idx )const
	{
		return layerData[idx];
	}
	void add( const LayerData& theLayerData )
	{
		layerData.push_back( theLayerData );
	}

private:
	std::vector<LayerData> layerData;


};
