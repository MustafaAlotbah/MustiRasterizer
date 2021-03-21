#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



namespace mge {





	struct vector2d {
		float x = 0;
		float y = 0;

		__device__ __host__ vector2d();
		__device__ __host__ vector2d(float x, float y);

		__device__ __host__ bool operator ==(const vector2d& p) const;
		__device__ __host__ bool operator !=(const vector2d& p) const;

		__device__ __host__ vector2d operator + (const float i) const;
		__device__ __host__ vector2d operator + (const vector2d p) const;
		__device__ __host__ vector2d& operator += (const float i);
		__device__ __host__ vector2d& operator += (const vector2d p);

		__device__ __host__ vector2d  operator - (const float i) const;
		__device__ __host__ vector2d  operator - (const vector2d p) const;
		__device__ __host__ vector2d& operator -= (const float i);
		__device__ __host__ vector2d& operator -= (const vector2d p);


		__device__ __host__ vector2d  operator * (const float i) const;
		__device__ __host__ vector2d  operator * (const vector2d p) const;
		__device__ __host__ vector2d& operator *= (const float i);
		__device__ __host__ vector2d& operator *= (const vector2d p);

		__device__ __host__ vector2d  operator / (const float i) const;
		__device__ __host__ vector2d  operator / (const vector2d p) const;
		__device__ __host__ vector2d& operator /= (const float i);
		__device__ __host__ vector2d& operator /= (const vector2d p);

		__device__ __host__ vector2d conjugate() const;

	};






	/*
	a11 a12
	a21 a22
	*/
	struct matrix2d {
		float a11 = 1;
		float a12 = 0;
		float a21 = 0;
		float a22 = 1;

		__device__ __host__ matrix2d();
		__device__ __host__ matrix2d(float a11, float a12, float a21, float a22);

		/*	__device__ __host__ bool operator ==(const matrix2d& p) const;
			__device__ __host__ bool operator !=(const matrix2d& p) const;

			__device__ __host__ matrix2d operator + (const float i) const;
			__device__ __host__ matrix2d operator + (const matrix2d p) const;
			__device__ __host__ matrix2d& operator += (const float i);
			__device__ __host__ matrix2d& operator += (const matrix2d p);

			__device__ __host__ matrix2d  operator - (const float i) const;
			__device__ __host__ matrix2d  operator - (const matrix2d p) const;
			__device__ __host__ matrix2d& operator -= (const float i);
			__device__ __host__ matrix2d& operator -= (const matrix2d p);
			*/

			/*__device__ __host__ matrix2d  operator * (const float i) const;
			__device__ __host__ matrix2d  operator * (const matrix2d p) const;*/

	//	__device__ __host__ vector2d  operator * (const vector2d p) const;
		__device__ __host__ vector2d  operator * (const vector2d p) const {
			return vector2d(a11 * p.x + a12 * p.y , a21 * p.x + a22 * p.y);
		}

		/*	__device__ __host__ matrix2d& operator *= (const float i);
			__device__ __host__ matrix2d& operator *= (const matrix2d p);

			__device__ __host__ matrix2d  operator / (const float i) const;
			__device__ __host__ matrix2d  operator / (const matrix2d p) const;
			__device__ __host__ matrix2d& operator /= (const float i);
			__device__ __host__ matrix2d& operator /= (const matrix2d p);*/


	};




}