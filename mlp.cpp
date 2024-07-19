#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

class MLP : public torch::nn::Module{

	public:
		MLP( int in_features, int out_features );
		torch::Tensor forward( torch::Tensor x );
	private:
		torch::nn::Linear ln{nullptr};
		torch::nn::BatchNorm1d bn{nullptr};

};

MLP::MLP( int in_features, int out_features ){
	ln = register_module( "ln", torch::nn::Linear( torch::nn::LinearOptions( in_features, out_features )));
	bn = register_module( "bn", torch::nn::BatchNorm1d( torch::nn::BatchNorm1dOptions( out_features )));
}

torch::Tensor MLP::forward( torch::Tensor x ){
	x = torch::relu(ln->forward(x));
	x = bn(x);
	return x;
}

torch::Tensor toyData2D( int samples ){
	auto ret = torch::randn(samples);
	return ret;
}

int main (int argc, char *argv[]) {
	
	std::cout << toyData2D(10) << std::endl;

	return 0;
}
