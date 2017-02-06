#pragma once

#include "MatrixContainer.h"
#include <vector>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace MathNet::Numerics::LinearAlgebra::Single;

public ref class ManagedMatrixContainer
{
public:
	ManagedMatrixContainer() : ManagedMatrixContainer(false)
	{		
	}

	ManagedMatrixContainer(bool rowMajor)
		: _container(new MatrixContainer()),
		_handles(gcnew List<GCHandle>()),
		_rowMajor(rowMajor)
	{		
	}

	~ManagedMatrixContainer()
	{
		delete _container;

		for (int i = 0; i < _handles->Count; i++)
		{
			_handles[i].Free();
		}
	}

	void AddMatrix(Matrix^ matrix)
	{
		auto arr = _rowMajor ? matrix->ToRowMajorArray() : matrix->ToColumnMajorArray();
		auto handle = GCHandle::Alloc(arr, GCHandleType::Pinned);
		_handles->Add(handle);
		_container->AddMatrix(matrix->RowCount, matrix->ColumnCount, (float*)(void*)handle.AddrOfPinnedObject());
	}

	std::vector<RawMatrixPtr*>& matrices() { return _container->matrices(); }
private:
	MatrixContainer*	_container;
	List<GCHandle>^		_handles;
	bool _rowMajor;
};
