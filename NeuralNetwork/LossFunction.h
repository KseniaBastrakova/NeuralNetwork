#pragma once
#include <vector>

class NeuralNetwork;

double GetError( const NeuralNetwork& net, const std::vector<std::vector<double>>& inputValues,
				 const std::vector<std::vector<double>>& expectedResults );

double GetAccuracyPercent( const NeuralNetwork& net, const std::vector<std::vector<double>>& inputValues,
				 const std::vector<std::vector<double>>& expectedResults );