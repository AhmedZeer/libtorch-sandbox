#include <ATen/core/TensorBody.h>
#include <ATen/ops/rand.h>
#include <c10/util/Optional.h>
#include <torch/data/dataloader.h>
#include <torch/data/datasets/base.h>
#include <torch/data/transforms/stack.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <vector>

class CustomClass : public torch::data::datasets::Dataset<CustomClass>{
	
	public:

		CustomClass( const std::vector<torch::Tensor> & data, const std::vector<torch::Tensor> & target ) :
			data_(data), target_(target) {}

		torch::data::Example<> get( size_t index ) override {
			return { data_[index], target_[index] };
		}

		torch::optional<size_t> size() const override {
			return data_.size();
		}
	
	private:
		std::vector<torch::Tensor> data_;
		std::vector<torch::Tensor> target_;

};

int main (int argc, char *argv[]) {

	std::vector< torch::Tensor > data;
	std::vector< torch::Tensor > target;

	for( int i = 0; i < 64; i++ ){
		data.push_back(torch::rand({10}));
		target.push_back(torch::tensor( i % 10 ));
	}

	auto dataset = CustomClass(data,target).map(torch::data::transforms::Stack<>());
	auto dataloader = torch::data::make_data_loader(dataset, 16);

	for( auto& batch : *dataloader ){

		auto data = batch.data;
		auto target = batch.target;

		std::cout << "X: " << data << "\n";
		std::cout << "y: " << target << "\n";

	}
	
	return 0;
}
