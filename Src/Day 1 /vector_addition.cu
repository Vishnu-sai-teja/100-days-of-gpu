#include <stdio.h>

__global__
void vector_addition_kernel(float *A, float *B, float *C, int n){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
  if (i < n){
    C[i] = A[i] + B[i]; 
  }
}

__host__
void vector_addition_host(float *A, float *B, float *C, int n){
  // Create the varaiables inside the device and copy the values from host to device
  float *A_d, *B_d, *C_d;
  int size = n * sizeof(float);
  cudaMalloc((void **) &A_d, size);
  cudaMalloc((void **) &B_d, size);
  cudaMalloc((void **) &C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  // process vector vector_addition
  vector_addition_kernel <<<int(ceil(n/256.0)), 256>>>(A_d, B_d, C_d, n);

  // copy back from device to host and free the memory
  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  for(int i=0; i<5; i++){
    printf("C at %d position is %f\n", i, C[i]);
  }
}
int main(){
  float *A, *B, *C;

  float X[5] = {1, 2, 3, 4, 5};
  float Y[5] = {6, 7, 8, 9, 10};
  float Z[5] = {};

  A = &X[0];
  B = &Y[0];
  C = &Z[0];
  vector_addition_host(A, B, C, 5);
}