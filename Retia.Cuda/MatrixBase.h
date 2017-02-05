#pragma once

template <class TIterator>
class MatrixBase
{
public:
	MatrixBase(int rows, int columns, int seqLength)
		: _rows(rows),
		_columns(columns),
		_seqLength(seqLength)
	{
		_length = _rows * _columns * seqLength;
	}

	virtual ~MatrixBase() = default;

	int rows() const
	{
		return _rows;
	}

	int columns() const
	{
		return _columns;
	}

	int seqLength() const
	{
		return _seqLength;
	}

	int length() const
	{
		return _length;
	}

	virtual TIterator begin() = 0;
	virtual TIterator end() = 0;
	virtual float* raw_ptr() = 0;

protected:
	int _rows = 0, _columns = 0, _seqLength = 0;
	int _length = 0;
};