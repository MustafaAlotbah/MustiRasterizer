#pragma once
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<initializer_list>

// cuda matrix

class cvector4d {
private:
	void* gpuAddr = 0;

public:
	__host__ ~cvector4d();
	__host__ cvector4d(const cvector4d& m);
	__host__ cvector4d(float i);
	__host__ cvector4d(float data[4]);
	void* getAddr() const;
	//
	float* read();
	__host__ cvector4d& operator=(const float i);

	__host__ cvector4d& operator=(const cvector4d&);
	////
	////
	__host__ cvector4d operator + (const cvector4d m) const;
	__host__ cvector4d& operator += (const cvector4d m);
	__host__ cvector4d operator + (const float i) const;
	__host__ cvector4d& operator += (const float i);
	////
	__host__ cvector4d  operator - (const cvector4d m) const;
	__host__ cvector4d& operator -= (const cvector4d m);
	__host__ cvector4d  operator - (const float i) const;
	__host__ cvector4d& operator -= (const float i);
	////
	////
	__host__ cvector4d  operator * (const cvector4d m) const;
	__host__ cvector4d& operator *= (const cvector4d m);
	__host__ cvector4d  operator * (const float i) const;
	__host__ cvector4d& operator *= (const float i);
	////
	////
	__host__ cvector4d  operator / (const float i) const;
	__host__ cvector4d& operator /= (const float i);

	//__host__ cvector4d  operator * (const cmatrix4d m) const;
	//__host__ cvector4d& operator *= (const cmatrix4d m);


};




class cmatrix4d {
private:
	void* gpuAddr = 0;

public:
	// __host__ cmatrix4d();
	__host__ ~cmatrix4d();
	__host__ cmatrix4d(const cmatrix4d& m);
	__host__ cmatrix4d(float i);
	__host__ cmatrix4d(float data[4][4]);

	float** read();
	__host__ cmatrix4d& operator=(const float i);

	__host__ cmatrix4d& operator=(const cmatrix4d&);
	//
	//__host__ bool operator ==(const cmatrix4d& m) const;
	//__host__ bool operator !=(const cmatrix4d& m) const;
	//
	__host__ cmatrix4d operator + (const cmatrix4d m) const;
	__host__ cmatrix4d& operator += (const cmatrix4d m);
	__host__ cmatrix4d operator + (const float i) const;
	__host__ cmatrix4d& operator += (const float i);
	//
	__host__ cmatrix4d  operator - (const cmatrix4d m) const;
	__host__ cmatrix4d& operator -= (const cmatrix4d m);
	__host__ cmatrix4d  operator - (const float i) const;
	__host__ cmatrix4d& operator -= (const float i);
	//
	//
	__host__ cmatrix4d  operator * (const cmatrix4d m) const;
	__host__ cmatrix4d& operator *= (const cmatrix4d m);
	__host__ cmatrix4d  operator * (const float i) const;
	__host__ cmatrix4d& operator *= (const float i);
	//
	//
	__host__ cmatrix4d  operator / (const float i) const;
	__host__ cmatrix4d& operator /= (const float i);

	__host__ cvector4d  operator * (const cvector4d m) const;
	// __host__ cvector4d& operator *= (const cvector4d m);
	//

};
