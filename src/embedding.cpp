#include <ATen/ops/embedding.h>
#include <iostream>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/options/embedding.h>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
  auto embed =
      torch::nn::Embedding(torch::nn::EmbeddingOptions(3, 3).max_norm(1));

  std::cout << embed->weight << std::endl;
  /*std::cout << embed << std::endl;*/

  return 0;
}
