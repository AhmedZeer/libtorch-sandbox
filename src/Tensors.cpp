#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/one_hot.h>
#include <ATen/ops/quantile.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <iostream>
#include <vector>


int main (int argc, char *argv[]) {

	// A Tensor full of 0s.
	auto t = torch::zeros({2,3});
	std::cout << "Zero Tensor: \n" << t << std::endl;

	// A Tensor full of 1s.
	t = torch::ones({2,3});
	std::cout << "Ones Tensor: \n" << t << std::endl;

	// (I)dentiry Matrix.
	t = torch::eye(3);
	std::cout << "Eye Tensor: \n" << t << std::endl;

	// A Tensor full of X.
	t = torch::full({2,3},3);
	std::cout << "Full Tensor: \n" << t << std::endl;

	// Create a tensor from std::vector< >.data() or a list.
	std::vector<float> vec = {1,2,3};
	auto t2 = torch::from_blob(vec.data(), {(int)vec.size()}, torch::kFloat);
	std::cout << "Blob Tensor: \n" << t2 << std::endl;

	// Clone,
	t = t2.clone();
	std::cout << "Cloned Tensor: \n" << t << std::endl;

	// Use other tendsors.
	t = torch::empty_like(t2);
	std::cout << "Cloned Tensor: \n" << t << std::endl;

	t = torch::full({10},3);
	t = t.view({2,5});
	std::cout << "View Tensor: \n" << t << std::endl;

	t = torch::full({10},3);
	t = t.view({2,5});
	t = t.transpose(0,1);
	std::cout << "Transposed View Tensor: \n" << t << std::endl;

	t = torch::rand({5,2});
	t = t.view({1,2,-1});
	t = t.permute({1,0,2});
	std::cout << "Transposed View Tensor: \n" << t << std::endl;

	auto b = torch::rand({5,3,32,32});
	std::cout << b.sizes() << std::endl;
	std::cout << b.index_select(0,torch::tensor({4})).sizes() << std::endl;

	auto c = torch::rand({3,4});
	auto mask = torch::zeros({3,4});
	// mask[0][0] = 1;

	std::cout << c << std::endl;
	std::cout << c.index( {mask.to(torch::kBool)} )<< std::endl;

	return 0;
}

