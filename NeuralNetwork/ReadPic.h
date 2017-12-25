#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
using namespace std;

typedef unsigned char uchar;

std::vector<std::vector<double>> read_mnist_images( string full_path, int current )
{
	std::vector<std::vector<double>> vec;
	auto reverseInt = [] ( int i )
	{
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = ( i >> 8 ) & 255, c3 = ( i >> 16 ) & 255, c4 = ( i >> 24 ) & 255;
		return ( ( int ) c1 << 24 ) + ( ( int ) c2 << 16 ) + ( ( int ) c3 << 8 ) + c4;
	};

	int number_of_images;

	ifstream file( full_path, ios::binary );

	if ( file.is_open() )
	{
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read( ( char * ) &magic_number, sizeof( magic_number ) );
		magic_number = reverseInt( magic_number );

		if ( magic_number != 2051 ) throw runtime_error( "Invalid MNIST image file!" );

		file.read( ( char * ) &number_of_images, sizeof( number_of_images ) ), number_of_images = reverseInt( number_of_images );
		file.read( ( char * ) &n_rows, sizeof( n_rows ) ), n_rows = reverseInt( n_rows );
		file.read( ( char * ) &n_cols, sizeof( n_cols ) ), n_cols = reverseInt( n_cols );

		int	image_size = n_rows * n_cols;

		uchar** _dataset = new uchar*[number_of_images];

		for ( int i = 0; i < std::min( number_of_images, current ); i++ )
		{
			std::vector<double> tp;
			for ( int r = 0; r < n_rows; ++r )
			{
				for ( int c = 0; c < n_cols; ++c )
				{
					unsigned char temp = 0;
					file.read( ( char* ) &temp, sizeof( temp ) );
					tp.push_back( ( double ) temp / 255.0 );
				}
			}
			vec.push_back( tp );
		}
	}
	else
	{
		throw runtime_error( "Cannot open file `" + full_path + "`!" );
	}
	return vec;
}


std::vector<double> read_mnist_labels( string full_path, size_t current )
{
	std::vector<double> vec;
	auto reverseInt = [] ( int i )
	{
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = ( i >> 8 ) & 255, c3 = ( i >> 16 ) & 255, c4 = ( i >> 24 ) & 255;
		return ( ( int ) c1 << 24 ) + ( ( int ) c2 << 16 ) + ( ( int ) c3 << 8 ) + c4;
	};

	typedef unsigned char uchar;
	int number_of_labels;
	ifstream file( full_path, ios::binary );

	if ( file.is_open() )
	{
		int magic_number = 0;
		file.read( ( char * ) &magic_number, sizeof( magic_number ) );
		magic_number = reverseInt( magic_number );

		if ( magic_number != 2049 ) throw runtime_error( "Invalid MNIST label file!" );

		file.read( ( char * ) &number_of_labels, sizeof( number_of_labels ) ), number_of_labels = reverseInt( number_of_labels );

		for ( size_t i = 0; i <std::min((size_t)number_of_labels, current); i++ )
		{
			unsigned char temp = 0;
			file.read( ( char* ) &temp, sizeof( temp ) );
			vec.push_back( ( double ) temp );
		}
	}
	else
	{
		throw runtime_error( "Unable to open file `" + full_path + "`!" );
	}
	return vec;
}
