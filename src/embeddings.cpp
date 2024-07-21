#include <ATen/ops/linear.h>
#include <ATen/ops/tensor.h>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/base.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/options/embedding.h>
#include <torch/nn/options/linear.h>
#include <torch/torch.h>

class CBOW : public torch::nn::Module{

	public:
		CBOW( int vocab_size, int embed_size = 300 );
		torch::Tensor forward( torch::Tensor x );
	
	private:
		torch::nn::Embedding embedd;
		torch::nn::Linear ln{nullptr};
};

CBOW::CBOW( int vocab_size, int embed_size ){

	embedd = torch::nn::Embedding( torch::nn::EmbeddingOptions(vocab_size, embed_size).max_norm(1) );;
	ln = torch::nn::Linear( torch::nn::LinearOptions( embed_size, vocab_size ) );

	register_module("embedd", embedd);
	register_module("ln", ln);

}

torch::Tensor CBOW::forward( torch::Tensor x ){

	x = embedd->forward(x);
	x = ln->forward(x);

	return x;
}

class my_data : public torch::data::datasets::Dataset<torch::Tensor>{
};

int main (int argc, char *argv[]) {
	

	auto data = my_data();
	torch::data::make_data_loader( x, torch::data::DataLoaderOptions().batch_size(200));
	return 0;
}
