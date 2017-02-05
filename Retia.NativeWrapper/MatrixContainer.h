#pragma once

#include <vector>
#include <memory>
#include "RawMatrixPtr.h"

class MatrixContainer
{
public:

	~MatrixContainer()
	{
		for(int i = 0; i < _matrices.size(); i++)
		{
			delete _matrices[i];
		}
	}

	std::vector<RawMatrixPtr*>& matrices() { return _matrices; }

	void AddMatrix(int rows, int columns, float* ptr)
	{
		_matrices.push_back(new RawMatrixPtr(rows, columns, 1, ptr));
	}
private:
	std::vector<RawMatrixPtr*>	_matrices;
};
