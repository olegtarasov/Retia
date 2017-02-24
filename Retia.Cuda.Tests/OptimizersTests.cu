#include <gtest/gtest.h>
#include <iostream>
#include "Matrix.h"
#include "NeuroWeigth.h"
#include "Helpers.h"
#include "Algorithms.h"

using std::cout;
using std::endl;

float Rosenbrock(HostMatrix& matrix)
{
	auto ptr = matrix.raw_ptr();
	return 100 * pow(ptr[1] - pow(ptr[0], 2), 2) + pow(1 - ptr[0], 2);
}

void RosenbrockGrad(HostMatrix& input, HostMatrix& grad)
{
	auto ptrIn = input.raw_ptr();
	auto ptrGrad = grad.raw_ptr();

	ptrGrad[0] = -400 * (ptrIn[1] - pow(ptrIn[0], 2)) * ptrIn[0] - 2 * (1 - ptrIn[0]);
	ptrGrad[1] = 200 * (ptrIn[1] - pow(ptrIn[0], 2));
}

TEST(RMSPropTests, CanOptimizeSimple)
{
	auto target = HostMatrix(2, 2, 1);
	auto weight = HostMatrix(2, 2, 1);
	auto grad = HostMatrix(2, 2, 1);
	auto cache1 = HostMatrix(2, 2, 1);
	auto cache2 = HostMatrix(2, 2, 1);
	auto cacheM = HostMatrix(2, 2, 1);

	InitMatrix(target, new float[4]{ 10.0f, 20.0f, -30.0f, 0.0f });
	weight.ZeroMemory();
	grad.ZeroMemory();
	cache1.ZeroMemory();
	cache2.ZeroMemory();
	cacheM.ZeroMemory();

	//PrintWeights(weight, grad, cache1, cache2, cacheM, -1);
	for (int i = 0; i < 200; ++i)
	{
		Algorithms::PropagateError(weight, target, grad);
		Algorithms::RMSPropOptimize(weight, grad, cache1, cache2, cacheM, 0.01f, 0.95f, 0.9f, 0.0f);
		//PrintWeights(weight, grad, cache1, cache2, cacheM, i);
	}

	auto tPtr = target.raw_ptr();
	auto wPtr = weight.raw_ptr();

	for (int i = 0; i < target.length(); ++i)
	{
		ASSERT_NEAR(wPtr[i], tPtr[i], 0.05);
	}
}

TEST(RMSPropTests, CanOptimizeRosenbrock)
{
	auto weight = HostMatrix(2, 1, 1);
	auto grad = HostMatrix(2, 1, 1);
	auto cache1 = HostMatrix(2, 1, 1);
	auto cache2 = HostMatrix(2, 1, 1);
	auto cacheM = HostMatrix(2, 1, 1);

	InitMatrix(weight, new float[2]{ 0, 0 });
	grad.ZeroMemory();
	cache1.ZeroMemory();
	cache2.ZeroMemory();
	cacheM.ZeroMemory();

	//cout << "Rosenbrock: " << Rosenbrock(weight) << endl;
	//PrintWeights(weight, grad, cache1, cache2, cacheM, -1);
	for (int i = 0; i < 10000; ++i)
	{
		RosenbrockGrad(weight, grad);
		Algorithms::RMSPropOptimize(weight, grad, cache1, cache2, cacheM, 5e-4f, 0.99f, 0.0f, 0.0f);
		//cout << "Rosenbrock: " << Rosenbrock(weight) << endl;
		//PrintWeights(weight, grad, cache1, cache2, cacheM, i);
	}

	auto wPtr = weight.raw_ptr();

	ASSERT_NEAR(Rosenbrock(weight), 0.0f, 1e-4f);
	for (int i = 0; i < weight.length(); ++i)
	{
		ASSERT_NEAR(wPtr[i], 1.0f, 1e-2f);
	}

	//cout << "Rosenbrock: " << Rosenbrock(weight) << endl;
}



