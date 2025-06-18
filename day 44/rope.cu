#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/types.h>

#define BLOCK_SIZE 256
#define theta 10000.0f
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

__global__ void rope_kernel(float* x, float* out, int N){ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x_v = reinterpret_cast<float4*>(&(x[idx * 4]))[0];

  int token_pos = idx / N; 
  int token_idx = idx % N;
  
  float exp_f_v = 1.0f / powf(theta, token_idx * 2 / (N * 4));
  float exp_s_v = 1.0f / powf(theta, ((token_idx * 2) + 1) / (N * 4));
  
  float sin_f_v = sinf(token_pos / exp_f_v);
  float cos_f_v = cosf(token_pos / exp_f_v);
  
  float sin_s_v = sinf(token_pos / exp_s_v);
  float cos_s_v = cosf(token_pos / exp_s_v);
  float4 out_v;

  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v; 
  
  reinterpret_cast<float4*>(&(out[idx * 4]))[0] = out_v;
}

void rope(torch::Tensor x, torch::Tensor out) {
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  
  int N = (int)(hidden_size/4);

  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);

  rope_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rope)
}