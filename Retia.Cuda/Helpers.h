#pragma once

#include <iostream>
#include "NeuroWeigth.h"

template <class TMatrix>
void InitMatrix(TMatrix& matrix, float *array)
{
	auto ptr = matrix.raw_ptr();
	for (int i = 0; i < matrix.length(); ++i)
	{
		ptr[i] = array[i];
	}
}

template <class TMatrix>
void PrintMatrix(TMatrix& matrix)
{
	for (int seq = 0; seq < matrix.seqLength(); ++seq)
	{
		for (int row = 0; row < matrix.rows(); ++row)
		{
			for (int col = 0; col < matrix.columns(); ++col)
			{
				std::cout << matrix.get_slow(row, col, seq) << '\t';
			}

			std::cout << std::endl;
		}
		
		std::cout << std::endl << " " << std::endl;
	}
}

template <class TMatrix>
void PrintWeights(TMatrix& weight, TMatrix& gradient, TMatrix& cache1, TMatrix& cache2, TMatrix& cacheM, int iteration)
{
	std::cout << "Iteration " << iteration << ". Weights: " << std::endl;
	PrintMatrix(weight);
	std::cout << "Grad:" << std::endl;
	PrintMatrix(gradient);
	std::cout << "Cache1:" << std::endl;
	PrintMatrix(cache1);
	std::cout << "Cache2:" << std::endl;
	PrintMatrix(cache2);
	std::cout << "CacheM:" << std::endl;
	PrintMatrix(cacheM);
	std::cout << " " << std::endl << "=====================" << std::endl << " " << std::endl;
}

inline void PrintWeights(NeuroWeigth& weight)
{
	PrintWeights(weight.weight(), weight.gradient(), weight.cache1(), weight.cache2(), weight.cache_m(), 0);
}