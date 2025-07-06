#include <torch/extension.h>
#include "ATen/ATen.h"

void RMSV1(float *input, float *output, int w, int h);

torch::Tensor RMS_V1(torch::Tensor input)
{
    auto out = torch::empty_like(input);
    int h = input.size(0);
    int w = input.size(1);
    RMSV1(input.data_ptr<float>(), out.data_ptr<float>(), w, h);
    return out;
}

void RMSV2(float *input, float *output, int w, int h);
torch::Tensor RMS_V2(torch::Tensor input)
{
    auto out = torch::empty_like(input);
    int h = input.size(0);
    int w = input.size(1);
    RMSV1(input.data_ptr<float>(), out.data_ptr<float>(), w, h);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("RMSV1", &RMS_V1, "RMSV1 (CUDA)");
    m.def("RMSV2", &RMS_V2, "RMSV2 (CUDA)");
}