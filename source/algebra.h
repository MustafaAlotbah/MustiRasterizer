#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


namespace mge {





	struct vector2d {
		float x = 0;
		float y = 0;

		__device__ __host__ vector2d();
		__device__ __host__ vector2d(float i);
		__device__ __host__ vector2d(float x, float y);

		__device__ __host__ bool operator ==(const vector2d& p) const;
		__device__ __host__ bool operator !=(const vector2d& p) const;

		__device__ __host__ vector2d& operator = (const float i) ;
		__device__ __host__ vector2d& operator = (const vector2d& p) ;

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
		__device__ __host__ matrix2d(float i);
		__device__ __host__ matrix2d(float a11, float a12, float a21, float a22);

		__device__ __host__ matrix2d& operator = (const float i);
		__device__ __host__ matrix2d& operator = (const matrix2d& p);
		__device__ __host__ bool operator ==(const matrix2d& p) const;
		__device__ __host__ bool operator !=(const matrix2d& p) const;

		__device__ __host__ matrix2d operator + (const float i) const;
		__device__ __host__ matrix2d operator + (const matrix2d p) const;
		__device__ __host__ matrix2d& operator += (const float i);
		__device__ __host__ matrix2d& operator += (const matrix2d p);

		__device__ __host__ matrix2d  operator - (const float i) const;
		__device__ __host__ matrix2d  operator - (const matrix2d p) const;
		__device__ __host__ matrix2d& operator -= (const float i);
		__device__ __host__ matrix2d& operator -= (const matrix2d p);
			

		__device__ __host__ matrix2d  operator * (const float i) const;
		__device__ __host__ matrix2d  operator * (const matrix2d p) const;

		__device__ __host__ vector2d  operator * (const vector2d p) const {
			return vector2d(a11 * p.x + a12 * p.y , a21 * p.x + a22 * p.y);
		}

		__device__ __host__ matrix2d& operator *= (const float i);
		__device__ __host__ matrix2d& operator *= (const matrix2d p);

		__device__ __host__ matrix2d  operator / (const float i) const;
		__device__ __host__ matrix2d& operator /= (const float i);


	};









	struct vector3d {
		float x = 0;
		float y = 0;
		float z = 0;

		__device__ __host__ vector3d();
		__device__ __host__ vector3d(float i);
		__device__ __host__ vector3d(float x, float y, float z);

		__device__ __host__ bool operator ==(const vector3d& p) const;
		__device__ __host__ bool operator !=(const vector3d& p) const;

		__device__ __host__ vector3d& operator = (const float i);
		__device__ __host__ vector3d& operator = (const vector3d& p);

		__device__ __host__ vector3d operator + (const float i) const;
		__device__ __host__ vector3d operator + (const vector3d p) const;
		__device__ __host__ vector3d& operator += (const float i);
		__device__ __host__ vector3d& operator += (const vector3d p);

		__device__ __host__ vector3d  operator - (const float i) const;
		__device__ __host__ vector3d  operator - (const vector3d p) const;
		__device__ __host__ vector3d& operator -= (const float i);
		__device__ __host__ vector3d& operator -= (const vector3d p);


		__device__ __host__ vector3d  operator * (const float i) const;
		__device__ __host__ vector3d  operator * (const vector3d p) const;
		__device__ __host__ vector3d& operator *= (const float i);
		__device__ __host__ vector3d& operator *= (const vector3d p);

		__device__ __host__ vector3d  operator / (const float i) const;
		__device__ __host__ vector3d  operator / (const vector3d p) const;
		__device__ __host__ vector3d& operator /= (const float i);
		__device__ __host__ vector3d& operator /= (const vector3d p);

		__device__ __host__ operator vector2d() const { return vector2d(x, y); }

		__device__ __host__ float abs() const {
			return (x * x + y * y + z * z);
		}

		__device__ __host__ operator float() const { return abs(); }


	};












	// not generalizing the dimensionality to avoid copying arrays

	struct matrix3d {
		float a11 = float(1), a12 = float(0), a13 = float(0);
		float a21 = float(0), a22 = float(1), a23 = float(0);
		float a31 = float(0), a32 = float(0), a33 = float(1);

	public:
		__device__ __host__ matrix3d();
		__device__ __host__ matrix3d(float i);
		__device__ __host__ matrix3d(float data[3][3]);


		__device__ __host__ matrix3d& operator=(const float i);

		__device__ __host__ matrix3d& operator=(const matrix3d& p);

		__device__ __host__ bool operator ==(const matrix3d& m) const;
		__device__ __host__ bool operator !=(const matrix3d& m) const;

		__device__ __host__ matrix3d operator + (const matrix3d m) const;
		__device__ __host__ matrix3d& operator += (const matrix3d m);
		__device__ __host__ matrix3d operator + (const float i) const;
		__device__ __host__ matrix3d& operator += (const float i);

		__device__ __host__ matrix3d  operator - (const matrix3d m) const;
		__device__ __host__ matrix3d& operator -= (const matrix3d m);
		__device__ __host__ matrix3d  operator - (const float i) const;
		__device__ __host__ matrix3d& operator -= (const float i);


		__device__ __host__ matrix3d  operator * (const matrix3d m) const;
		__device__ __host__ matrix3d& operator *= (const matrix3d m);
		__device__ __host__ matrix3d  operator * (const float i) const;
		__device__ __host__ matrix3d& operator *= (const float i);


		__device__ __host__ matrix3d  operator / (const float i) const;
		__device__ __host__ matrix3d& operator /= (const float i);

		__device__ __host__ vector3d  operator * (const vector3d m) const;
		__device__ __host__ vector3d& operator *= (const vector3d m);


	};






	struct vector4d {
		float x = 0;
		float y = 0;
		float z = 0;
		float w = 0;

		__device__ __host__ vector4d();
		__device__ __host__ vector4d(float i);
		__device__ __host__ vector4d(float x, float y, float z, float w);

		__device__ __host__ bool operator ==(const vector4d& p) const;
		__device__ __host__ bool operator !=(const vector4d& p) const;

		__device__ __host__ vector4d& operator = (const float i);
		__device__ __host__ vector4d& operator = (const vector4d& p);

		__device__ __host__ vector4d operator + (const float i) const;
		__device__ __host__ vector4d operator + (const vector4d p) const;
		__device__ __host__ vector4d& operator += (const float i);
		__device__ __host__ vector4d& operator += (const vector4d p);

		__device__ __host__ vector4d  operator - (const float i) const;
		__device__ __host__ vector4d  operator - (const vector4d p) const;
		__device__ __host__ vector4d& operator -= (const float i);
		__device__ __host__ vector4d& operator -= (const vector4d p);


		__device__ __host__ vector4d  operator * (const float i) const;
		__device__ __host__ vector4d  operator * (const vector4d p) const;
		__device__ __host__ vector4d& operator *= (const float i);
		__device__ __host__ vector4d& operator *= (const vector4d p);

		__device__ __host__ vector4d  operator / (const float i) const;
		__device__ __host__ vector4d  operator / (const vector4d p) const;
		__device__ __host__ vector4d& operator /= (const float i);
		__device__ __host__ vector4d& operator /= (const vector4d p);

		__device__ __host__ operator vector2d() const { return vector2d(x, y); }
		__device__ __host__ operator vector3d() const { return vector3d(x, y, z); }

		__device__ __host__ float abs() const {
			return (x * x + y * y + z * z + w * w);
		}

		__device__ __host__ operator float() const { return abs(); }


	};









	struct matrix4d {
		float a11 = float(1), a12 = float(0), a13 = float(0), a14 = float(0);
		float a21 = float(0), a22 = float(1), a23 = float(0), a24 = float(0);
		float a31 = float(0), a32 = float(0), a33 = float(1), a34 = float(0);
		float a41 = float(0), a42 = float(0), a43 = float(0), a44 = float(1);

	public:
		__device__ __host__ matrix4d();
		__device__ __host__ matrix4d(float i);
		__device__ __host__ matrix4d(float data[4][4]);


		__device__ __host__ matrix4d& operator=(const float i);

		__device__ __host__ matrix4d& operator=(const matrix4d& p);

		__device__ __host__ bool operator ==(const matrix4d& m) const;
		__device__ __host__ bool operator !=(const matrix4d& m) const;

		__device__ __host__ matrix4d operator + (const matrix4d m) const;
		__device__ __host__ matrix4d& operator += (const matrix4d m);
		__device__ __host__ matrix4d operator + (const float i) const;
		__device__ __host__ matrix4d& operator += (const float i);

		__device__ __host__ matrix4d  operator - (const matrix4d m) const;
		__device__ __host__ matrix4d& operator -= (const matrix4d m);
		__device__ __host__ matrix4d  operator - (const float i) const;
		__device__ __host__ matrix4d& operator -= (const float i);


		__device__ __host__ matrix4d  operator * (const matrix4d m) const;
		__device__ __host__ matrix4d& operator *= (const matrix4d m);
		__device__ __host__ matrix4d  operator * (const float i) const;
		__device__ __host__ matrix4d& operator *= (const float i);


		__device__ __host__ matrix4d  operator / (const float i) const;
		__device__ __host__ matrix4d& operator /= (const float i);

		__device__ __host__ vector4d  operator * (const vector4d m) const;
		__device__ __host__ vector4d& operator *= (const vector4d m);


	};




















}