#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif



template <typename scalar_t>
__global__ void sort_attributes_forward_kernel(
    const scalar_t* __restrict__ point_cloud,
    scalar_t* __restrict__ in_attributes,
    scalar_t* __restrict__ out_grid_attributes,
    scalar_t* __restrict__ num_points_in_quad,
    size_t num_points,
    float kernel_radius,
    size_t kernel_size,
    size_t channels) {

  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  // const int index = j*num_points +i;


  if(i != j){
      auto vec_X = point_cloud[j*3 + 0] - point_cloud[i*3 + 0];
      auto vec_Y = point_cloud[j*3 + 1] - point_cloud[i*3 + 1];
      auto vec_Z = point_cloud[j*3 + 2] - point_cloud[i*3 + 2];
      float mag = sqrt(pow(vec_X,2)+pow(vec_Y,2)+pow(vec_Z,2));
      if(mag < kernel_radius){
          float vox_radius = (float)kernel_radius/(float)kernel_size;
          int quad = 0;
          int num_quads = kernel_size *8;
          vec_X =  (vec_X <= 0)? 0: 1;
          vec_Y =  (vec_Y <= 0)? 0: 2;
          if (vec_Z >= 0){
              quad = vec_X + vec_Y;
          }
          else{
              quad = vec_X + vec_Y + 4;
          }

          quad = quad + floor(mag/vox_radius)*8;

          for(int c = 0; c < channels; c++){
            atomicAdd(&out_grid_attributes[c + (quad * channels) + (i * num_quads * channels)],
                        in_attributes[c + j * channels]);
            }

            atomicAdd(&num_points_in_quad[quad + i * num_quads],1.0);

      }

  }

}






template <typename scalar_t>
__global__ void sort_in_grid_kernel(
    const scalar_t* __restrict__ point_cloud,
    scalar_t* __restrict__ quads,
    size_t num_points,
    float kernel_radius,
    size_t kernel_size) {

  const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int j = (blockIdx.y * blockDim.y) + threadIdx.y;
  const int index = j*num_points +i;


  if(i == j){
      quads[index]= -1;
  }
  else{
      auto vec_X = point_cloud[j*3 + 0] - point_cloud[i*3 + 0];
      auto vec_Y = point_cloud[j*3 + 1] - point_cloud[i*3 + 1];
      auto vec_Z = point_cloud[j*3 + 2] - point_cloud[i*3 + 2];
      float mag = sqrt(pow(vec_X,2)+pow(vec_Y,2)+pow(vec_Z,2));
      if(mag < kernel_radius){
          float vox_radius = (float)kernel_radius/(float)kernel_size;
          int cell_num = 0;
          vec_X =  (vec_X <= 0)? 0: 1;
          vec_Y =  (vec_Y <= 0)? 0: 2;
          if (vec_Z >= 0){
              cell_num = vec_X + vec_Y;
          }
          else{
              cell_num = vec_X + vec_Y + 4;
          }

          cell_num = cell_num + floor(mag/vox_radius)*8;

          quads[index]= cell_num;

      }
      else{
          quads[index]= -2;

      }

  }

}



















at::Tensor sort_attributes_cuda_forward(
    at::Tensor in_attributes,
    at::Tensor point_cloud,
    at::Tensor out_grid_attributes,//(2048,quads,c_in)
    at::Tensor num_points_in_quad,//(2048,quads,c_in)
    float kernel_radius,
    float kernel_size,
    int channels) {



        // printf("kernel radius %f \n",kernel_radius);

        const auto num_points = point_cloud.size(0);
        // at::Tensor z;
        // const auto num_quads = kernel_size*8;
        // at::zeros_out(z,{num_points,num_quads});

        // auto num_points_in_quad = at::zeros_like(point_cloud);
        // at::resize(num_points_in_quad,{num_points,num_quads});
        // num_points_in_quad.resize_({num_points,num_quads});


        const dim3 threadsPerBlock(32, 32);  // 1024 threads
        // const dim3 numBlocks(64,64);
        const dim3 numBlocks(num_points/threadsPerBlock.x,  /* for instance 512/8 = 64*/
            num_points/threadsPerBlock.y);


// printf("here1");

        AT_DISPATCH_FLOATING_TYPES(point_cloud.type(), "sort_in_grid_forward_cuda", ([&] {
        sort_attributes_forward_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            point_cloud.data<scalar_t>(),
            in_attributes.data<scalar_t>(),
            out_grid_attributes.data<scalar_t>(),
            num_points_in_quad.data<scalar_t>(),
            num_points,
            kernel_radius,
            kernel_size,
            channels);
            }));

            num_points_in_quad.clamp_min_(1);
            num_points_in_quad.unsqueeze_(2);
            out_grid_attributes = out_grid_attributes/num_points_in_quad;
    // printf("here2");
  return out_grid_attributes;
}





at::Tensor sort_in_grid_cuda(
    at::Tensor point_cloud,
    at::Tensor quads,
    float kernel_radius,
    float kernel_size) {


        // // assert foo is 2-dimensional and holds floats.
        // auto point_cloud_a = point_cloud.accessor<float,2>();
        // auto quads_a = quads.accessor<float,2>();


        // printf("kernel radius %f \n",kernel_radius);

        const auto num_points = point_cloud.size(0);
        const dim3 threadsPerBlock(32, 32);  // 1024 threads
        // const dim3 numBlocks(64,64);
        const dim3 numBlocks(num_points/threadsPerBlock.x,  /* for instance 512/8 = 64*/
            num_points/threadsPerBlock.y);


// printf("here1");

        AT_DISPATCH_FLOATING_TYPES(point_cloud.type(), "sort_in_grid_forward_cuda", ([&] {
        sort_in_grid_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            point_cloud.data<scalar_t>(),
            quads.data<scalar_t>(),
            num_points,
            kernel_radius,
            kernel_size);
            }));
    // printf("here2");
  return quads;
}



        // auto X = at::cat({old_h, input}, /*dim=*/1);
        // auto gates = at::addmm(bias, X, weights.transpose(0, 1));
        //
        // const auto batch_size = old_cell.size(0);
        //
        //
        // auto new_h = at::zeros_like(old_cell);
        // auto new_cell = at::zeros_like(old_cell);
        // auto input_gate = at::zeros_like(old_cell);
        // auto output_gate = at::zeros_like(old_cell);
        // auto candidate_cell = at::zeros_like(old_cell);
