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
	ManagedMatrixContainer()
		: _container(new MatrixContainer()),
		_handles(gcnew List<GCHandle>())
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
		auto handle = GCHandle::Alloc(matrix->AsColumnMajorArray(), GCHandleType::Pinned);
		_handles->Add(handle);
		_container->AddMatrix(matrix->RowCount, matrix->ColumnCount, (float*)(void*)handle.AddrOfPinnedObject());
	}

	std::vector<RawMatrixPtr*>& matrices() { return _container->matrices(); }
private:
	MatrixContainer*	_container;
	List<GCHandle>^		_handles;
};
