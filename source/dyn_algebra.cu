#include "dyn_algebra.h"


void* cvector4d::getAddr() const {
	return gpuAddr;
}

cvector4d::cvector4d(float i) {
	// allocate memory on GPU

	float _data[4] = { 0 };
	for (int n = 0; n < 4; n++)
		_data[n ] = i;
	cudaMalloc((void**)&gpuAddr, 4 * sizeof(float));
	cudaMemcpy(gpuAddr, _data, 4 * sizeof(float), cudaMemcpyHostToDevice);
}


cvector4d::cvector4d(float data[4]) {
	// allocate memory on GPU
	cudaMalloc((void**)&gpuAddr, 4 * sizeof(float));
	// copy values to GPU
	cudaMemcpy(gpuAddr, data, 4 * sizeof(float), cudaMemcpyHostToDevice);
}
//
// copy constructor important so that cmatrix4d's can be returned
cvector4d::cvector4d(const cvector4d& m) {
	// allocate memory
	cudaMalloc((void**)&gpuAddr, 4 * sizeof(float));
	// copy values
	cudaMemcpy(gpuAddr, m.gpuAddr, 4 * sizeof(float), cudaMemcpyDeviceToDevice);
}
//
cvector4d::~cvector4d() {
	// Free memory
	cudaFree(gpuAddr);
}
//
//
//
float* cvector4d::read() {

	float* res = new float[16];
	cudaError_t r = cudaMemcpy(res, gpuAddr, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	
	return res;
}

cvector4d& cvector4d::operator=(const float i) {
	float _data[4] = { 0 };
	for (int n = 0; n < 4; n++)
		_data[n] = i;
	cudaMalloc((void**)&gpuAddr, 4 * sizeof(float));
	cudaMemcpy(gpuAddr, _data, 4 * sizeof(float), cudaMemcpyHostToDevice);
	return *this;
}


__host__ cvector4d& cvector4d::operator=(const cvector4d& m) {
	// allocate memory
	cudaMalloc((void**)&gpuAddr, 4 * sizeof(float));
	// copy values
	cudaMemcpy(gpuAddr, m.gpuAddr, 4 * sizeof(float), cudaMemcpyDeviceToDevice);
	return *this;
}



__global__ void vector4addvector4Kernel(float* res, const float* a, const float* b)
{
	int i = threadIdx.x;
	res[i] = a[i] + b[i];
}

cvector4d cvector4d::operator + (const cvector4d m) const {
	cvector4d result(4);
	vector4addvector4Kernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	return result;
}


cvector4d& cvector4d::operator += (const cvector4d m) {
	vector4addvector4Kernel << <1, 4 >> > ((float*)gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	return *this;
}

__global__ void vector4addfloatKernel(float* res, const float* a, float b)
{
	int i = threadIdx.x;
	res[i] = a[i] + b;
}

cvector4d cvector4d::operator + (const float i) const {
	cvector4d result(4);
	vector4addfloatKernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)gpuAddr, i);
	return result;
}


cvector4d& cvector4d::operator += (const float b) {
	vector4addfloatKernel << <1, 4 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	return *this;
}




__global__ void vector4subvector4Kernel(float* res, const float* a, const float* b)
{
	int i = threadIdx.x;
	res[i] = a[i] - b[i];
}

cvector4d cvector4d::operator - (const cvector4d m) const {
	cvector4d result(4);
	vector4subvector4Kernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	return result;
}


cvector4d& cvector4d::operator -= (const cvector4d m) {
	vector4subvector4Kernel << <1, 4 >> > ((float*)gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	return *this;
}

__global__ void vector4subfloatKernel(float* res, const float* a, float b)
{
	int i = threadIdx.x;
	res[i] = a[i] - b;
}

cvector4d cvector4d::operator - (const float i) const {
	cvector4d result(4);
	vector4subfloatKernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)gpuAddr, i);
	return result;
}


cvector4d& cvector4d::operator -= (const float b) {
	vector4subfloatKernel << <1, 4 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	return *this;
}



__global__ void vector4_mult_vector4Kernel(float* res, const float* a, const float* b)
{
	int i = threadIdx.x;
	res[i] = a[i] * b[i];
}

cvector4d cvector4d::operator * (const cvector4d m) const {
	cvector4d result(4);
	vector4_mult_vector4Kernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	return result;
}


cvector4d& cvector4d::operator *= (const cvector4d m) {
	vector4_mult_vector4Kernel << <1, 4 >> > ((float*)gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	return *this;
}

__global__ void vector4_mult_floatKernel(float* res, const float* a, float b)
{
	int i = threadIdx.x;
	res[i] = a[i] * b;
}

cvector4d cvector4d::operator * (const float i) const {
	cvector4d result(4);
	vector4_mult_floatKernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)gpuAddr, i);
	return result;
}


cvector4d& cvector4d::operator *= (const float b) {
	vector4_mult_floatKernel << <1, 4 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	return *this;
}



__global__ void vector4_div_floatKernel(float* res, const float* a, float b)
{
	int i = threadIdx.x;
	res[i] = a[i] / b;
}

cvector4d cvector4d::operator / (const float i) const {
	cvector4d result(4);
	vector4_div_floatKernel << <1, 4 >> > ((float*)result.gpuAddr, (float*)gpuAddr, i);
	return result;
}


cvector4d& cvector4d::operator /= (const float b) {
	vector4_div_floatKernel << <1, 4 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	return *this;
}











/////////// MATRIX 4D ///////////////













cmatrix4d::cmatrix4d(float i) {
	// allocate memory on GPU

	float _data[16] = { 0 };
	for (int n = 0; n < 4; n++)
		_data[n*4+n] = i;
	cudaMalloc((void**)&gpuAddr, 16 * sizeof(float));
	cudaMemcpy(gpuAddr, _data, 16 * sizeof(float), cudaMemcpyHostToDevice);
	
}


cmatrix4d::cmatrix4d(float data[4][4]) {
	float _data[16] = { 0 };
	for (int i = 0; i < 16; i++)
		_data[i] = data[i / 4][i % 4];
	// allocate memory on GPU
	cudaMalloc((void**)&gpuAddr, 16 * sizeof(float));
	// copy values to GPU
	cudaMemcpy(gpuAddr, _data, 16 * sizeof(float), cudaMemcpyHostToDevice);
}

// copy constructor important so that cmatrix4d's can be returned
cmatrix4d::cmatrix4d(const cmatrix4d& m) {
	// allocate memory
	cudaMalloc((void**)&gpuAddr, 16 * sizeof(float));
	// copy values
	cudaMemcpy(gpuAddr, m.gpuAddr, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
}

cmatrix4d::~cmatrix4d() {
	// Free memory
	cudaFree(gpuAddr);
}



float** cmatrix4d::read() {

	float* res = new float[16];
	cudaError_t r = cudaMemcpy(res, gpuAddr, 16 * sizeof(float), cudaMemcpyDeviceToHost);
	float** fres = new float* [4];
	for (int i = 0; i < 4; i++)
	{
		fres[i] = new float[4];
		for (int j = 0; j < 4; j++)
		{
			fres[i][j] = res[i * 4 + j];
		}
	}
	return fres;
}


 cmatrix4d& cmatrix4d::operator=(const float i) {
	 float _data[16] = { 0 };
	 for (int n = 0; n < 4; n++)
		 _data[n * 4 + n] = i;
	 cudaMalloc((void**)&gpuAddr, 16 * sizeof(float));
	 cudaMemcpy(gpuAddr, _data, 16 * sizeof(float), cudaMemcpyHostToDevice);
	 return *this;
}


 __host__ cmatrix4d& cmatrix4d::operator=(const cmatrix4d& m) {
	 // allocate memory
	 cudaMalloc((void**)&gpuAddr, 16 * sizeof(float));
	 // copy values
	 cudaMemcpy(gpuAddr, m.gpuAddr, 16 * sizeof(float), cudaMemcpyDeviceToDevice);
	 return *this;
 }



 __global__ void matrix4addmatrix4Kernel(float* res, const float* a, const float* b)
 {
	 int i = threadIdx.x;
	 res[i] = a[i] + b[i];
 }

 cmatrix4d cmatrix4d::operator + (const cmatrix4d m) const {
	 cmatrix4d result(16);
	 matrix4addmatrix4Kernel <<<1, 16 >>> ((float*)result.gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	 return result;
 }


 cmatrix4d& cmatrix4d::operator += (const cmatrix4d m) {
	 matrix4addmatrix4Kernel << <1, 16 >> > ((float*)gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	 return *this;
 }

 __global__ void matrix4addfloatKernel(float* res, const float* a, float b)
 {
	 int i = threadIdx.x;
	 res[i] = a[i] + b;
 }

 cmatrix4d cmatrix4d::operator + (const float i) const {
	 cmatrix4d result(16);
	 matrix4addfloatKernel << <1, 16 >> > ((float*)result.gpuAddr, (float*)gpuAddr, i);
	 return result;
 }


 cmatrix4d& cmatrix4d::operator += (const float b) {
	 matrix4addfloatKernel << <1, 16 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	 return *this;
 }



 __global__ void matrix4submatrix4Kernel(float* res, const float* a, const float* b)
 {
	 int i = threadIdx.x;
	 res[i] = a[i] - b[i];
 }


 cmatrix4d cmatrix4d::operator - (const cmatrix4d m) const {
	 cmatrix4d result(16);
	 matrix4submatrix4Kernel << <1, 16 >> > ((float*)result.gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	 return result;
 }


 cmatrix4d& cmatrix4d::operator -= (const cmatrix4d m) {
	 matrix4submatrix4Kernel << <1, 16 >> > ((float*)gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	 return *this;
 }

 __global__ void matrix4subfloatKernel(float* res, const float* a, float b)
 {
	 int i = threadIdx.x;
	 res[i] = a[i] - b;
 }



 cmatrix4d cmatrix4d::operator - (const float b) const {
	 cmatrix4d result(16);
	 matrix4subfloatKernel << <1, 16 >> > ((float*)result.gpuAddr, (float*)gpuAddr, b);
	 return result;
 }


 cmatrix4d& cmatrix4d::operator -= (const float b) {
	 matrix4subfloatKernel << <1, 16 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	 return *this;
 }


 __global__ void matrix4_mult_floatKernel(float* res, float* a, float b)
 {
	 int i = threadIdx.x;
	 res[i] = a[i] * b;
 }

  cmatrix4d  cmatrix4d::operator * (const float b) const {
	 cmatrix4d result(16);
	 matrix4_mult_floatKernel << <1, 16 >> > ((float*)result.gpuAddr, (float*)gpuAddr, b);
	 return result;
 }
  cmatrix4d& cmatrix4d::operator *= (const float b) {
	 matrix4_mult_floatKernel << <1, 16 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	 return *this;
 }


 __global__ void matrix4_div_floatKernel(float* res, const float* a, float b)
 {
	 int i = threadIdx.x;
	 res[i] = a[i] / b;
 }

 __host__ cmatrix4d  cmatrix4d::operator / (const float b) const {
	 cmatrix4d result(16);
	 matrix4_div_floatKernel << <1, 16 >> > ((float*)result.gpuAddr, (float*)gpuAddr, b);
	 return result;
 }
 __host__ cmatrix4d& cmatrix4d::operator /= (const float b) {
	 matrix4_div_floatKernel << <1, 16 >> > ((float*)gpuAddr, (float*)gpuAddr, b);
	 return *this;
 }


 __global__ void matrix4_mult_matrix4Kernel(float* res, const float* a, float* b)
 {
	 int id = threadIdx.x;
	 int i = id % 4;
	 int j = id / 4;
	 res[id] = 0;
	 for (int k = 0; k < 4; k++)
	 {
		 res[id] += a[i + k * 4] * b[k + j * 4];
	 }
 }


 cmatrix4d cmatrix4d::operator * (const cmatrix4d m) const {
	 cmatrix4d result(16);
	 matrix4_mult_matrix4Kernel << <1, 16 >> > ((float*)result.gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	 return result;
 }


 cmatrix4d& cmatrix4d::operator *= (const cmatrix4d m) {
	 matrix4_mult_matrix4Kernel << <1, 16 >> > ((float*)gpuAddr, (float*)m.gpuAddr, (float*)gpuAddr);
	 return *this;
 }


 __global__ void matrix4_mult_vector4Kernel(float* res, const float* mat, float* vec)
 {
	 int id = threadIdx.x;
	 res[id] = 0;
	 for (int k = 0; k < 4; k++)
	 {
		 res[id] += mat[k + id * 4] * vec[k];
	 }
 }


 cvector4d cmatrix4d::operator * (const cvector4d m) const {
	 cvector4d result(.0f);
	 matrix4_mult_vector4Kernel << <1, 4 >> > ((float*)result.getAddr(), (float*)gpuAddr, (float*)m.getAddr());
	 return result;
 }