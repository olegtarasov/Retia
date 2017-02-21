#pragma once
#include "AlgorithmTester.h"
#include "ManagedMatrixContainer.h"

using namespace MathNet::Numerics::LinearAlgebra;

namespace Retia::NativeWrapper {
	public ref class MathTester
	{
	public:

		static double TestCrossEntropyError(Matrix<float>^ output, Matrix<float>^ target)
		{
			auto matrices = gcnew ManagedMatrixContainer();
			matrices->AddMatrix(output);
			matrices->AddMatrix(target);

			double result = AlgorithmTester::CrossEntropyError(*matrices->matrices()[0], *matrices->matrices()[1]);

			delete matrices;

			return result;
		}

		static void TestCrossEntropyBackpropagation(Matrix<float>^ output, Matrix<float>^ target, Matrix<float>^ result)
		{
			auto matrices = gcnew ManagedMatrixContainer();
			matrices->AddMatrix(output);
			matrices->AddMatrix(target);
			matrices->AddMatrix(result);

			AlgorithmTester::BackpropagateCrossEntropy(*matrices->matrices()[0], *matrices->matrices()[1], *matrices->matrices()[2]);

			delete matrices;
		}

		static void RMSPropOptimize(Matrix<float>^ weight, Matrix<float>^ gradient, Matrix<float>^ cache1, Matrix<float>^ cache2, Matrix<float>^ cacheM,
			float learningRate, float decayRate, float momentum, float weightDecay)
		{
			auto matrices = gcnew ManagedMatrixContainer();
			matrices->AddMatrix(weight);
			matrices->AddMatrix(gradient);
			matrices->AddMatrix(cache1);
			matrices->AddMatrix(cache2);
			matrices->AddMatrix(cacheM);

			AlgorithmTester::RMSPropOptimize(*matrices->matrices()[0], *matrices->matrices()[1], *matrices->matrices()[2], *matrices->matrices()[3], *matrices->matrices()[4],
				learningRate, decayRate, momentum, weightDecay);
		}

		static void TestClampMatrix(Matrix<float>^ matrix, float value)
		{
			auto matrices = gcnew ManagedMatrixContainer();
			matrices->AddMatrix(matrix);

			AlgorithmTester::ClampMatrix(*matrices->matrices()[0], value);

			delete matrices;
		}
	};
}