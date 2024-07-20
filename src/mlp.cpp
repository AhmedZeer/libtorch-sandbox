#include <ATen/core/TensorBody.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/concat.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/sum.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>

// The Backbone Of Our NN.
class LinearBnReluImpl : public torch::nn::Module{

	public:
		LinearBnReluImpl( int in_features, int out_features );
		torch::Tensor forward( torch::Tensor x );
	private:
		torch::nn::Linear ln{nullptr};
		torch::nn::BatchNorm1d bn{nullptr};

};

// Defining The Constructor.
LinearBnReluImpl::LinearBnReluImpl( int in_features, int out_features ){
	ln = register_module( "ln", torch::nn::Linear( torch::nn::LinearOptions( in_features, out_features )));
	bn = register_module( "bn", torch::nn::BatchNorm1d( torch::nn::BatchNorm1dOptions( out_features )));
}

// Output.
torch::Tensor LinearBnReluImpl::forward( torch::Tensor x ){
	x = torch::relu(ln->forward(x));
	x = bn(x);
	return x;
}

// Mark This Class As A Module.
TORCH_MODULE(LinearBnRelu);

class MLP : public torch::nn::Module{

	public:
		MLP( int in_feature, int out_features );
		torch::Tensor forward( torch::Tensor x );

	private:
		int hidden[3] = {32,64,128};
		LinearBnRelu ln1{nullptr};
		LinearBnRelu ln2{nullptr};
		LinearBnRelu ln3{nullptr};
		torch::nn::Linear out_ln{nullptr};
};

// MLP Implementation.
MLP::MLP( int in_features, int out_features ){

	ln1 = LinearBnRelu( in_features, hidden[0] );
	ln2 = LinearBnRelu( hidden[0], hidden[1] );
	ln3 = LinearBnRelu( hidden[1], hidden[2] );
	out_ln = torch::nn::Linear( hidden[2], out_features );

	ln1 = register_module("ln1", ln1);
	ln2 = register_module("ln2", ln2);
	ln3 = register_module("ln3", ln3);
	out_ln = register_module("out_ln", out_ln);
}

torch::Tensor MLP::forward( torch::Tensor x ){
	x = ln1->forward(x);
	x = ln2->forward(x);
	x = ln3->forward(x);
	x = out_ln->forward(x);
	return x;
}

// Generate a Linear toy data to test the NN.
// TODO : Add a dim parameter -> For loop the current Algo.
torch::Tensor toyData2D( int samples ){

	auto ret_x = torch::arange(samples).to( torch::kFloat );
	auto ret_y = torch::arange(samples).to( torch::kFloat );
	
	auto ret = torch::stack({ret_x,ret_y},0);
	auto noise = torch::randn(samples);

	ret[0].add_(noise);
	ret[1].sub_(noise);

	return ret;
}

int main (int argc, char *argv[]) {
	
	auto mlp = MLP(10,1);

	auto input  = torch::rand({2,10});
	auto target = torch::ones({2,1});
	torch::optim::Adam optim_mlp( mlp.parameters(), 0.001 );
	
	for( int i = 0; i < 100; i++ ){
		optim_mlp.zero_grad();
		auto pred = mlp.forward(input);
		auto cost = torch::mse_loss(pred, target);
		cost.backward();
		optim_mlp.step();
		std::cout << cost.item() << std::endl;
	}

	return 0;
}
