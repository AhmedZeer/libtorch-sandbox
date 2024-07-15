#include <ATen/core/TensorBody.h>
#include <ATen/ops/quantile.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <torch/torch.h>
#include <iostream>

int main (int argc, char *argv[]) {
	auto t = torch::zeros({2,3});
	std::cout << t << std::endl;
	return 0;
}

