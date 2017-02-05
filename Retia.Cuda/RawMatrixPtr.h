#pragma once
#include "MatrixBase.h"

class RawMatrixPtr : public MatrixBase<float*>
{
public:
	RawMatrixPtr(int rows, int columns, int seqLength, float* rawPtr)
		: MatrixBase(rows, columns, seqLength)
	{
		_ptr = rawPtr;
	}

	float* begin() override { return _ptr; }
	float* end() override { return _ptr + _length; }
	float* raw_ptr() override { return _ptr; }
private:
	float*	_ptr;
};
