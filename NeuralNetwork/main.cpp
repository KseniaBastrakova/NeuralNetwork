#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "NeuralNetwork.h"
#include "ReadPic.h"
#include "LearningAlgorithm.h"
#include "SerializationFunction.h"

const int numberNetLearnElements = 1000000;

int main( int argc, char** argv )
{
	if ( argc < 5 )
	{
		std::cout << "Error input_args: " << std::endl;
		std::cout << "1: Path to MNIST train-images " << std::endl;
		std::cout << "2: Path to MNIST train-labels " << std::endl;
		std::cout << "3: Path to MNIST test-images " << std::endl;
		std::cout << "4: Path to MNIST test-labels " << std::endl;
		std::cout << "5: number hidden  layers (default = 1) " << std::endl;
		std::cout << "6: learnRate (default = 0.9) " << std::endl;
		std::cout << "7: crossError stop in train (default 0.001) " << std::endl;
		return 0;
	}

	//std::string trainImageMNIST( "C:\\test\\train-images-idx3-ubyte\\train-images.idx3-ubyte" );
//	std::string trainLabelsMNIST( "C:\\test\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte" );
	
//	std::string testImageMNIST( "C:\\test\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte" );
//	std::string testLabelsMNIST( "C:\\test\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte" );

	std::string trainImageMNIST( argv[1] );
	std::string trainLabelsMNIST( argv[2] );
	std::string testImageMNIST( argv[3] );
	std::string testLabelsMNIST( argv[4] );
	int hiddenLayerSize = 1;
	double learnRate = 0.9;
	double crossError = 0.001;



	switch ( argc )
	{
		case 6:
			hiddenLayerSize = atoi( argv[5] );
			break;
		case 7:
			hiddenLayerSize = atoi( argv[5] );
			learnRate = atof( argv[6] );
			break;
		case 8:
			hiddenLayerSize = atoi( argv[5] );
			learnRate = atof( argv[6] );
			crossError = atof( argv[7] );
			break;
	}

	int sizeInput = 784;
	int sizeOutput = 10;
	std::vector<int> hiddenLayersSizes;
	hiddenLayersSizes.resize( hiddenLayerSize );


	for ( int i = 0; i < hiddenLayerSize; i++ )
	{
		cout << "Input size for hidden layer #   " << i<<"  ";
		cin >> hiddenLayersSizes[i];
	}
	std::vector<std::vector<double>> trainData_MNIST;
	trainData_MNIST = read_mnist_images( trainImageMNIST, numberNetLearnElements );

	std::vector<double> vec_labels = read_mnist_labels( trainLabelsMNIST, trainData_MNIST.size() );


	std::vector<std::vector<double>> trainLabel_MNIST;
	trainLabel_MNIST.resize( vec_labels.size() );
	for ( int i = 0; i < trainLabel_MNIST.size(); i++ )
	{
		trainLabel_MNIST[i] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		trainLabel_MNIST[i][( int ) vec_labels[i]] = 1.0;
	}



	std::vector<std::vector<double>> testData_MNIST;

	testData_MNIST = read_mnist_images( testImageMNIST,numberNetLearnElements );

	std::vector<double> vec_test_labels = read_mnist_labels( testLabelsMNIST, testData_MNIST.size());
	std::vector<std::vector<double>> testLabel_MNIST;
	testLabel_MNIST.resize( testData_MNIST.size() );
	for ( int i = 0; i < testLabel_MNIST.size(); i++ )
	{
		testLabel_MNIST[i] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		testLabel_MNIST[i][( int ) vec_test_labels[i]] = 1.0;
		
	}
	std::vector<size_t> size;
	size.resize( hiddenLayerSize + 2 );
	for ( int i = 1; i < hiddenLayerSize + 1; i++ )
		size[i] = hiddenLayersSizes[i - 1];
	size[0] = sizeInput;
	size[hiddenLayerSize + 1] = sizeOutput;
	NeuralNetwork netWork( size );

	Training trainNetwork( &netWork, trainData_MNIST, trainLabel_MNIST,crossError, learnRate );
	
	trainNetwork.Perform();

//	PrintNetwork( 3, size, netWork.GetLayers(), "C:\\test\\test.txt" );
//	NeuralNetwork* network2;

//	ReadNetwork( network2, "C:\\test\\test.txt" );

    double testError = netWork.TestTrain( testData_MNIST, testLabel_MNIST ) / testData_MNIST.size();

	double trainError = netWork.TestTrain( trainData_MNIST, trainLabel_MNIST ) / trainData_MNIST.size();

	std::cout << "!!!  result train  " << trainError << " result test " <<  testError<<std::endl;

	return 0;
}