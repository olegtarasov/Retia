#pragma once

#include "MatrixContainer.h"
#include <vector>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Runtime::InteropServices;
using namespace MathNet::Numerics::LinearAlgebra;
using namespace MathNet::Numerics::LinearAlgebra::Storage;

namespace Retia::NativeWrapper {

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
			if (rowMajor)
			{
				_arrayMap = gcnew List<Tuple<Matrix<float>^, array<float>^>^>();
			}
		}

		~ManagedMatrixContainer()
		{
			delete _container;

			for (int i = 0; i < _handles->Count; i++)
			{
				_handles[i].Free();
			}
		}

		void AddMatrix(Matrix<float>^ matrix)
		{
			// When we use AsColumnMajorArray, we get underlying matrix array.
			// When we use ToRowMajorArray(), we get a row-major copy and need to store the array for synchronization.
			array<float>^ arr;
			if (_rowMajor)
			{
				arr = matrix->ToRowMajorArray();
				_arrayMap->Add(gcnew Tuple<Matrix<float>^, array<float>^>(matrix, arr));
			}
			else
			{
				arr = matrix->AsColumnMajorArray();
			}

			auto handle = GCHandle::Alloc(arr, GCHandleType::Pinned);
			_handles->Add(handle);
			_container->AddMatrix(matrix->RowCount, matrix->ColumnCount, (float*)(void*)handle.AddrOfPinnedObject());
		}

		void SyncToMatrix()
		{
			if (!_rowMajor)
			{
				return;
			}

			for each (auto pair in _arrayMap)
			{
				auto storage = DenseColumnMajorMatrixStorage<float>::OfRowMajorArray(pair->Item1->RowCount, pair->Item1->ColumnCount, pair->Item2);
				storage->CopyTo(pair->Item1->Storage, MathNet::Numerics::LinearAlgebra::ExistingData::Clear);
			}
		}

		void SyncToArray()
		{
			if (!_rowMajor)
			{
				return;
			}

			for each (auto pair in _arrayMap)
			{
				auto newArr = pair->Item1->ToRowMajorArray();
				Array::Copy(newArr, 0, pair->Item2, 0, newArr->Length);
			}
		}

		std::vector<RawMatrixPtr*>& matrices() { return _container->matrices(); }
	private:
		MatrixContainer*	_container;
		List<GCHandle>^		_handles;
		bool				_rowMajor;

		List<Tuple<Matrix<float>^, array<float>^>^>^ _arrayMap = nullptr;
	};
}