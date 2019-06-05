#include <torch/torch.h>

#include <iostream>
#include <vector>


// CUDA forward declarations

at::Tensor sort_attributes_cuda_forward(
    at::Tensor in_attributes, //(2048,c_in)
    at::Tensor point_cloud, //(2048,3)
    at::Tensor out_grid_attributes, //(2048,quads,c_in)
    at::Tensor num_points_in_quad, //(2048,quads)
    float kernel_radius,
    float kernel_size,
    int channels);

// at::Tensor sort_in_grid_cuda_backward(
//     at::Tensor grad_output,
//     float kernel_size,
//     int channels);
//


at::Tensor sort_in_grid_cuda(
    at::Tensor point_cloud,
    at::Tensor quads,
    float kernel_radius,
    float kernel_size);







// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)





at::Tensor sort_attributes_forward(
    at::Tensor in_attributes,
    at::Tensor point_cloud,
    at::Tensor out_grid_attributes,//(2048,quads,c_in)
    at:: Tensor num_points_in_quad, // (2048,quads)
    float kernel_radius,
    float kernel_size,
    int channels) {
        CHECK_INPUT(point_cloud);
        CHECK_INPUT(in_attributes);
        CHECK_INPUT(out_grid_attributes);
        CHECK_INPUT(num_points_in_quad);

    // printf("here cpp ");
    return sort_attributes_cuda_forward(in_attributes,
                                    point_cloud,
                                    out_grid_attributes,
                                    num_points_in_quad,
                                    kernel_radius,
                                    kernel_size,
                                    channels);

}




at::Tensor sort_in_grid(
    at::Tensor point_cloud,
    at::Tensor quads,
    float kernel_radius,
    float kernel_size) {

    // printf("here cpp ");
    return sort_in_grid_cuda(point_cloud,quads,kernel_radius,kernel_size);

}




// at::Tensor sort_in_grid_backward(
//     at::Tensor grad_output,
//     float kernel_size,
//     int channels) {
//         CHECK_INPUT(grad_output);
//     // printf("here cpp ");
//     return sort_in_grid_cuda_backward(grad_output,kernel_size,channels);
//
// }
//






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sort_attributes_forward, "sort_in_grid forward (cuda)");
  m.def("output_sorted_points", &sort_in_grid, "sort_in_grid forward (cuda)");
  // m.def("backward", &sort_in_grid_backward, "sort_in_grid backward");
}
