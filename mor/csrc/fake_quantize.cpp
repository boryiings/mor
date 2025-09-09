#include <torch/extension.h>

#include <vector>

namespace mor {

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(mor, m) {
  m.def("fake_quantize(Tensor input, int dim_x, int dim_y, int block_x, int block_y, float clamp_threshold, int mode, int sf_type) -> Tensor[]");
  m.def("fake_quantize_no_alloc(Tensor input, int dim_x, int dim_y, int block_x, int block_y, float clamp_threshold, int mode, int sf_type, Tensor result) -> Tensor");
  m.def("quant_dequant_e4m3(Tensor input, int dim_x, int dim_y, int block_x, int block_y) -> Tensor");
  m.def("quant_dequant_e5m2(Tensor input, int dim_x, int dim_y, int block_x, int block_y) -> Tensor");
  m.def("quant_dequant_e3m2(Tensor input, int dim_x, int dim_y, int block_x, int block_y) -> Tensor");
  m.def("quant_dequant_e2m3(Tensor input, int dim_x, int dim_y, int block_x, int block_y) -> Tensor");
  m.def("e4m3_with_e8_scale_rne(Tensor input, int dim_x, int dim_y, int block_x, int block_y) -> Tensor");
  m.def("e4m3_with_e8_scale_rz(Tensor input, int dim_x, int dim_y, int block_x, int block_y) -> Tensor");
  m.def("fake_quantize_block_scaling(Tensor input, int dim_x, int dim_y, int block_x, int block_y, float e4m3_threshold, int ss_mode) -> Tensor[]");
  m.def("fake_quantize_block_scaling_inplace(Tensor input, int dim_x, int dim_y, int block_x, int block_y, float e4m3_threshold, int ss_mode) -> Tensor[]");
  m.def("fake_quantize_e8mx(Tensor input, int dim_x, int dim_y, int block_x, int block_y, int quantize_type, int scaling_strategy) -> Tensor");
  m.def("fake_quantize_e8mx_inplace(Tensor input, int dim_x, int dim_y, int block_x, int block_y, int quantize_type, int scaling_strategy, Tensor result) -> ()");
  m.def("fake_quantize_channel_scaling(Tensor input, int dim_x, int dim_y, int block_x, int block_y, float e4m3_threshold, int ss_mode) -> Tensor[]");
  m.def("fake_quantize_channel_scaling_inplace(Tensor input, int dim_x, int dim_y, int block_x, int block_y, float e4m3_threshold, int ss_mode) -> Tensor[]");
}

}
