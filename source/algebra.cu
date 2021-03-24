

#include "algebra.h"

namespace mge {

	// vector 2D Implementation


	__device__ __host__ vector2d::vector2d() {

	}


	__device__ __host__ vector2d& vector2d::operator = (const float i)  {
		this->x = i;
		this->y = i;
		return *this;
	}
	__device__ __host__ vector2d& vector2d::operator = (const vector2d& p)  {
		this->x = p.x;
		this->y = p.y;
		return *this;
	}

	__device__ __host__ vector2d::vector2d(float x, float y) {
		this->x = x;
		this->y = y;
	}
	__device__ __host__ vector2d::vector2d(float i) {
		this->x = i;
		this->y = i;
	}

	__device__ __host__ bool vector2d::operator ==(const vector2d& p) const {
		return this->x == p.x && this->y == p.y;
	}

	__device__ __host__ bool vector2d::operator !=(const vector2d& p) const{
		return this->x != p.x || this->y != p.y;
	}

	__device__ __host__ vector2d vector2d::operator + (const float i) const{
		return vector2d(this->x + i, this->y + i);
	}

	__device__ __host__ vector2d  vector2d::operator + (const vector2d p) const{
		return vector2d(this->x + p.x, this->y + p.y);
	}

	__device__ __host__ vector2d& vector2d::operator += (const float i){
		this->x += i;
		this->y += i;
		return *this;
	}

	__device__ __host__ vector2d& vector2d::operator += (const vector2d p){
		this->x += p.x;
		this->y += p.y;
		return *this;
	}

	__device__ __host__ vector2d  vector2d::operator - (const float i) const{
		return vector2d(this->x - i, this->y - i);
	}
	__device__ __host__ vector2d  vector2d::operator - (const vector2d p) const{
		return vector2d(this->x - p.x, this->y - p.y);
	}
	__device__ __host__ vector2d& vector2d::operator -= (const float i){
		this->x -= i;
		this->y -= i;
		return *this;
	}
	__device__ __host__ vector2d& vector2d::operator -= (const vector2d p){
		this->x -= p.x;
		this->y -= p.y;
		return *this;
	}

	__device__ __host__ vector2d  vector2d::operator * (const float i) const{
		return vector2d(
			this->x * i,
			this->y * i
		);
	}
	__device__ __host__ vector2d  vector2d::operator * (const vector2d p) const{
		return vector2d(
			this->x * p.x,
			this->y * p.y
		);
	}
	__device__ __host__ vector2d& vector2d::operator *= (const float i){
		this->x *= i;
		this->y *= i;
		return *this;
	}
	__device__ __host__ vector2d& vector2d::operator *= (const vector2d p){
		this->x *= p.x;
		this->y *= p.y;
		return *this;
	}

	__device__ __host__ vector2d  vector2d::operator / (const float i) const{
		return vector2d(
			this->x / i,
			this->y / i
		);
	}
	__device__ __host__ vector2d  vector2d::operator / (const vector2d p) const{
		return vector2d(
			this->x / p.x,
			this->y / p.y
		);
	}
	__device__ __host__ vector2d& vector2d::operator /= (const float i){
		this->x /= i;
		this->y /= i;
		return *this;
	}
	__device__ __host__ vector2d& vector2d::operator /= (const vector2d p){
		this->x /= p.x;
		this->y /= p.y;
		return *this;
	}

	__device__ __host__ vector2d vector2d::conjugate() const{
		return vector2d(
			this->x,
			-this->y
		);
	}




	// Matrix 2D Implementation
	__device__ __host__ matrix2d::matrix2d(){}
	__device__ __host__ matrix2d::matrix2d(float a11, float a12, float a21, float a22){
		this->a11 = a11;
		this->a12 = a12;
		this->a21 = a21;
		this->a22 = a22;
	}

	__device__ __host__ matrix2d::matrix2d(float i) {
		this->a11 = i;
		this->a12 = 0;
		this->a21 = 0;
		this->a22 = i;

	}


	__device__ __host__ matrix2d& matrix2d::operator = (const float i) {
		this->a11 = i;
		this->a12 = 0;
		this->a21 = 0;
		this->a22 = i;
		return *this;
	}
	__device__ __host__ matrix2d& matrix2d::operator = (const matrix2d& p) {
		this->a11 = p.a11;
		this->a12 = p.a12;
		this->a21 = p.a21;
		this->a22 = p.a22;
		return *this;
	}

	__device__ __host__ bool matrix2d::operator ==(const matrix2d& p) const{
		return (
			this->a11 == p.a11 &&
			this->a12 == p.a12 &&
			this->a21 == p.a21 &&
			this->a22 == p.a22
			);
	}

	__device__ __host__ bool matrix2d::operator !=(const matrix2d& p) const{
		return (
			this->a11 != p.a11 ||
			this->a12 != p.a12 ||
			this->a21 != p.a21 ||
			this->a22 != p.a22
			);
	}

	__device__ __host__ matrix2d  matrix2d::operator + (const float i) const{
		return matrix2d(
			this->a11 + i,
			this->a12 + i,
			this->a21 + i,
			this->a22 + i
		);
	}
	__device__ __host__ matrix2d  matrix2d::operator + (const matrix2d p) const{
		return matrix2d(
			this->a11 + p.a11,
			this->a12 + p.a12,
			this->a21 + p.a21,
			this->a22 + p.a22
		);
	}
	__device__ __host__ matrix2d& matrix2d::operator += (const float i){
		this->a11 += i;
		this->a12 += i;
		this->a21 += i;
		this->a22 += i;
		return *this;
	}

	__device__ __host__ matrix2d& matrix2d::operator += (const matrix2d p){
		this->a11 += p.a11;
		this->a12 += p.a12;
		this->a21 += p.a21;
		this->a22 += p.a22;
		return *this;
	}
	__device__ __host__ matrix2d  matrix2d::operator - (const float i) const{
		return matrix2d(
			this->a11 - i,
			this->a12 - i,
			this->a21 - i,
			this->a22 - i
		);
	}
	__device__ __host__ matrix2d  matrix2d::operator - (const matrix2d p) const{
		return matrix2d(
			this->a11 - p.a11,
			this->a12 - p.a12,
			this->a21 - p.a21,
			this->a22 - p.a22
		);
	}

	__device__ __host__ matrix2d& matrix2d::operator -= (const float i){
		this->a11 -= i;
		this->a12 -= i;
		this->a21 -= i;
		this->a22 -= i;
		return *this;
	}
	__device__ __host__ matrix2d& matrix2d::operator -= (const matrix2d p){
		this->a11 -= p.a11;
		this->a12 -= p.a12;
		this->a21 -= p.a21;
		this->a22 -= p.a22;
		return *this;
	}

	__device__ __host__ matrix2d  matrix2d::operator * (const matrix2d p) const{
		return matrix2d(
			this->a11 * p.a11 + this->a12 * p.a21,
			this->a11 * p.a12 + this->a12 * p.a22,
			this->a21 * p.a11 + this->a22 * p.a21,
			this->a21 * p.a12 + this->a22 * p.a22
		);
	}
	__device__ __host__ matrix2d& matrix2d::operator *= (const matrix2d p){
		this->a11 = this->a11 * p.a11 + this->a12 * p.a21;
		this->a12 = this->a11 * p.a12 + this->a12 * p.a22;
		this->a21 = this->a21 * p.a11 + this->a22 * p.a21;
		this->a22 = this->a21 * p.a12 + this->a22 * p.a22;
		return *this;
	}

	__device__ __host__ matrix2d  matrix2d::operator * (const float i) const {
		return matrix2d(
			this->a11 * i,
			this->a12 * i,
			this->a21 * i,
			this->a22 * i
		);
	}
	__device__ __host__ matrix2d& matrix2d::operator *= (const float i) {
		this->a11 *= i;
		this->a12 *= i;
		this->a21 *= i;
		this->a22 *= i;
		return *this;
	}

	__device__ __host__ matrix2d  matrix2d::operator / (const float i) const{
		return matrix2d(
			this->a11 / i,
			this->a12 / i,
			this->a21 / i,
			this->a22 / i
		);
	}
	__device__ __host__ matrix2d& matrix2d::operator /= (const float i){
		this->a11 /= i;
		this->a12 /= i;
		this->a21 /= i;
		this->a22 /= i;
		return *this;
	}






	// 3D vectors


	__device__ __host__ vector3d::vector3d() {

	}


	__device__ __host__ vector3d& vector3d::operator = (const float i) {
		this->x = i;
		this->y = i;
		this->z = i;
		return *this;
	}
	__device__ __host__ vector3d& vector3d::operator = (const vector3d& p) {
		this->x = p.x;
		this->y = p.y;
		this->z = p.z;
		return *this;
	}

	__device__ __host__ vector3d::vector3d(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
	__device__ __host__ vector3d::vector3d(float i) {
		this->x = i;
		this->y = i;
		this->z = i;
	}


	__device__ __host__ bool vector3d::operator ==(const vector3d& p) const {
		return this->x == p.x && this->y == p.y && this->z == p.z;
	}

	__device__ __host__ bool vector3d::operator !=(const vector3d& p) const {
		return this->x != p.x || this->y != p.y || this->z != p.z;
	}

	__device__ __host__ vector3d vector3d::operator + (const float i) const {
		return vector3d(this->x + i, this->y + i, this->z + i);
	}

	__device__ __host__ vector3d  vector3d::operator + (const vector3d p) const {
		return vector3d(this->x + p.x, this->y + p.y, this->z + p.z);
	}

	__device__ __host__ vector3d& vector3d::operator += (const float i) {
		this->x += i;
		this->y += i;
		this->z += i;
		return *this;
	}

	__device__ __host__ vector3d& vector3d::operator += (const vector3d p) {
		this->x += p.x;
		this->y += p.y;
		this->z += p.z;
		return *this;
	}

	__device__ __host__ vector3d  vector3d::operator - (const float i) const {
		return vector3d(this->x - i, this->y - i, this->z - i);
	}
	__device__ __host__ vector3d  vector3d::operator - (const vector3d p) const {
		return vector3d(this->x - p.x, this->y - p.y, this->z - p.z);
	}
	__device__ __host__ vector3d& vector3d::operator -= (const float i) {
		this->x -= i;
		this->y -= i;
		this->z -= i;
		return *this;
	}
	__device__ __host__ vector3d& vector3d::operator -= (const vector3d p) {
		this->x -= p.x;
		this->y -= p.y;
		this->y -= p.z;
		return *this;
	}

	__device__ __host__ vector3d  vector3d::operator * (const float i) const {
		return vector3d(
			this->x * i,
			this->y * i,
			this->z * i
		);
	}
	__device__ __host__ vector3d  vector3d::operator * (const vector3d p) const {
		return vector3d(
			this->x * p.x,
			this->y * p.y,
			this->z * p.z
		);
	}
	__device__ __host__ vector3d& vector3d::operator *= (const float i) {
		this->x *= i;
		this->y *= i;
		this->z *= i;
		return *this;
	}
	__device__ __host__ vector3d& vector3d::operator *= (const vector3d p) {
		this->x *= p.x;
		this->y *= p.y;
		this->z *= p.z;
		return *this;
	}

	__device__ __host__ vector3d  vector3d::operator / (const float i) const {
		return vector3d(
			this->x / i,
			this->y / i,
			this->z / i
		);
	}
	__device__ __host__ vector3d  vector3d::operator / (const vector3d p) const {
		return vector3d(
			this->x / p.x,
			this->y / p.y,
			this->z / p.z
		);
	}
	__device__ __host__ vector3d& vector3d::operator /= (const float i) {
		this->x /= i;
		this->y /= i;
		this->z /= i;
		return *this;
	}
	__device__ __host__ vector3d& vector3d::operator /= (const vector3d p) {
		this->x /= p.x;
		this->y /= p.y;
		this->z /= p.z;
		return *this;
	}







	// generic matrix implementation


	
	__device__ __host__ matrix3d::matrix3d(){
}

	
	__device__ __host__ matrix3d::matrix3d(float i){
		a11 = i;
		a22 = i;
		a33 = i;
	}

	
	__device__ __host__ matrix3d::matrix3d(float data[3][3]){
		a11 = data[0][0]; a12 = data[0][1]; a13 = data[0][2];
		a21 = data[1][0]; a22 = data[1][1]; a23 = data[1][2];
		a31 = data[2][0]; a32 = data[2][1]; a33 = data[2][2];
	}


	
	__device__ __host__ matrix3d& matrix3d::operator = (const float i) {
		a11 = i; a12 = 0; a13 = 0;
		a21 = 0; a22 = i; a23 = 0;
		a31 = 0; a32 = 0; a33 = i;
		return *this;
	}

	
	__device__ __host__ matrix3d& matrix3d::operator = (const matrix3d& m) {
		a11 = m.a11; a12 = m.a12; a13 = m.a13;
		a21 = m.a21; a22 = m.a22; a23 = m.a23;
		a31 = m.a31; a32 = m.a32; a33 = m.a33;
		return *this;
	}

	
	__device__ __host__ bool matrix3d::operator ==(const matrix3d& m) const {
		bool equal = true;
		equal &= a11 == m.a11 && a12 == m.a12 && a13 == m.a13;
		equal &= a21 == m.a21 && a22 == m.a22 && a23 == m.a23;
		equal &= a31 == m.a31 && a32 == m.a32 && a33 == m.a33;
		return equal;

	}

	
	__device__ __host__ bool matrix3d::operator !=(const matrix3d& m) const {
		bool equal = false;
		equal |= a11 != m.a11 || a12 != m.a12 || a13 != m.a13;
		equal |= a21 != m.a21 || a22 != m.a22 || a23 != m.a23;
		equal |= a31 != m.a31 || a32 != m.a32 || a33 != m.a33;
		return equal;

	}

	
	__device__ __host__ matrix3d  matrix3d::operator + (const matrix3d m) const {
		float temp[3][3] = {
				{	a11 + m.a11, a12 + m.a12, a13 + m.a13	},
				{	a21 + m.a21, a22 + m.a22, a23 + m.a23	},
				{	a31 + m.a31, a32 + m.a32, a33 + m.a33	}
		};
		return matrix3d(temp);
	}

	
	__device__ __host__ matrix3d& matrix3d::operator += (const matrix3d m) {
		a11 += m.a11; a12 += m.a12; a13 += m.a13;
		a21 += m.a21; a22 += m.a22; a23 += m.a23;
		a31 += m.a31; a32 += m.a32; a33 += m.a33;
		return *this;

	}

	
	__device__ __host__ matrix3d  matrix3d::operator + (const float i) const {
		float temp[3][3] = {
				{	a11 + i, a12 + i, a13 + i	},
				{	a21 + i, a22 + i, a23 + i	},
				{	a31 + i, a32 + i, a33 + i	}
		};
		return matrix3d(temp);

	}

	
	__device__ __host__ matrix3d& matrix3d::operator += (const float i) {
		a11 += i; a12 += i; a13 += i;
		a21 += i; a22 += i; a23 += i;
		a31 += i; a32 += i; a33 += i;
		return *this;
	}


	
	__device__ __host__ matrix3d  matrix3d::operator - (const matrix3d m) const{
		float temp[3][3] = {
				{	a11 - m.a11, a12 - m.a12, a13 - m.a13	},
				{	a21 - m.a21, a22 - m.a22, a23 - m.a23	},
				{	a31 - m.a31, a32 - m.a32, a33 - m.a33	}
		};
		return matrix3d(temp);
	}

	
	__device__ __host__ matrix3d& matrix3d::operator -= (const matrix3d m){
		a11 -= m.a11; a12 -= m.a12; a13 -= m.a13;
		a21 -= m.a21; a22 -= m.a22; a23 -= m.a23;
		a31 -= m.a31; a32 -= m.a32; a33 -= m.a33;
		return *this;
	}

	
	__device__ __host__ matrix3d  matrix3d::operator - (const float i) const{
		float temp[3][3] = {
				{	a11 - i, a12 - i, a13 - i	},
				{	a21 - i, a22 - i, a23 - i	},
				{	a31 - i, a32 - i, a33 - i	}
		};
		return matrix3d(temp);
	}

	
	__device__ __host__ matrix3d& matrix3d::operator -= (const float i){
		a11 -= i; a12 -= i; a13 -= i;
		a21 -= i; a22 -= i; a23 -= i;
		a31 -= i; a32 -= i; a33 -= i;
		return *this;
	}


	
	__device__ __host__ matrix3d  matrix3d::operator * (const matrix3d m) const {
		float temp[3][3] = {
				{	// first row
					a11 * m.a11 + a12 * m.a21 + a13 * m.a31, 
					a11 * m.a12 + a12 * m.a22 + a13 * m.a32,
					a11 * m.a13 + a12 * m.a23 + a13 * m.a33,
				},
				{	// second row		     
					a21 * m.a11 + a22 * m.a21 + a23 * m.a31,
					a21 * m.a12 + a22 * m.a22 + a23 * m.a32,
					a21 * m.a13 + a22 * m.a23 + a23 * m.a33,
				},
				{	// third row		     
					a31 * m.a11 + a32 * m.a21 + a33 * m.a31,
					a31 * m.a12 + a32 * m.a22 + a33 * m.a32,
					a31 * m.a13 + a32 * m.a23 + a33 * m.a33,
				}
		};
		return matrix3d(temp);
	}

	
	__device__ __host__ matrix3d& matrix3d::operator *= (const matrix3d m){
		// first row
		a11 = a11 * m.a11 + a12 * m.a21 + a13 * m.a31;
		a12 = a11 * m.a12 + a12 * m.a22 + a13 * m.a32;
		a13 = a11 * m.a13 + a12 * m.a23 + a13 * m.a33;

		// second row		     
		a21 = a21 * m.a11 + a22 * m.a21 + a23 * m.a31;
		a22 = a21 * m.a12 + a22 * m.a22 + a23 * m.a32;
		a23 = a21 * m.a13 + a22 * m.a23 + a23 * m.a33;

		// third row		     
		a31 = a31 * m.a11 + a32 * m.a21 + a33 * m.a31;
		a32 = a31 * m.a12 + a32 * m.a22 + a33 * m.a32;
		a33 = a31 * m.a13 + a32 * m.a23 + a33 * m.a33;
		return *this;
	}

	
	__device__ __host__ matrix3d  matrix3d::operator * (const float i) const{
		float temp[3][3] = {
				{	a11 * i, a12 * i, a13 * i	},
				{	a21 * i, a22 * i, a23 * i	},
				{	a31 * i, a32 * i, a33 * i	}
		};
		return matrix3d(temp);
	}

	
	__device__ __host__ matrix3d& matrix3d::operator *= (const float i) {
		a11 *= i; a12 *= i; a13 *= i;
		a21 *= i; a22 *= i; a23 *= i;
		a31 *= i; a32 *= i; a33 *= i;
		return *this;

	}


	
	__device__ __host__ matrix3d  matrix3d::operator / (const float i) const {
		float temp[3][3] = {
				{	a11 / i, a12 / i, a13 / i	},
				{	a21 / i, a22 / i, a23 / i	},
				{	a31 / i, a32 / i, a33 / i	}
		};
		return matrix3d(temp);
	}

	
	__device__ __host__ matrix3d& matrix3d::operator /= (const float i){
		a11 /= i; a12 /= i; a13 /= i;
		a21 /= i; a22 /= i; a23 /= i;
		a31 /= i; a32 /= i; a33 /= i;
		return *this;
	}




	__device__ __host__ vector3d  matrix3d::operator * (const vector3d v) const {
		float x = a11 * v.x + a12 * v.y + a13 * v.z;
		float y = a21 * v.x + a22 * v.y + a23 * v.z;
		float z = a31 * v.x + a32 * v.y + a33 * v.z;
		return vector3d(x, y, z);
	}








	// 4D vectors Quaternion


	__device__ __host__ vector4d::vector4d() {

	}


	__device__ __host__ vector4d& vector4d::operator = (const float i) {
		this->x = i;
		this->y = i;
		this->z = i;
		this->w = i;
		return *this;
	}
	__device__ __host__ vector4d& vector4d::operator = (const vector4d& p) {
		this->x = p.x;
		this->y = p.y;
		this->z = p.z;
		this->w = p.w;
		return *this;
	}

	__device__ __host__ vector4d::vector4d(float x, float y, float z, float w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
	__device__ __host__ vector4d::vector4d(float i) {
		this->x = i;
		this->y = i;
		this->z = i;
		this->w = i;
	}


	__device__ __host__ bool vector4d::operator ==(const vector4d& p) const {
		return this->x == p.x && this->y == p.y && this->z == p.z && this->w == p.w;
	}

	__device__ __host__ bool vector4d::operator !=(const vector4d& p) const {
		return this->x != p.x || this->y != p.y || this->z != p.z || this->w != p.w;
	}

	__device__ __host__ vector4d vector4d::operator + (const float i) const {
		return vector4d(this->x + i, this->y + i, this->z + i, this->w + i);
	}

	__device__ __host__ vector4d  vector4d::operator + (const vector4d p) const {
		return vector4d(this->x + p.x, this->y + p.y, this->z + p.z, this->w + p.w);
	}

	__device__ __host__ vector4d& vector4d::operator += (const float i) {
		this->x += i;
		this->y += i;
		this->z += i;
		this->w += i;
		return *this;
	}

	__device__ __host__ vector4d& vector4d::operator += (const vector4d p) {
		this->x += p.x;
		this->y += p.y;
		this->z += p.z;
		this->w += p.w;
		return *this;
	}

	__device__ __host__ vector4d  vector4d::operator - (const float i) const {
		return vector4d(this->x - i, this->y - i, this->z - i, this->w - i);
	}
	__device__ __host__ vector4d  vector4d::operator - (const vector4d p) const {
		return vector4d(this->x - p.x, this->y - p.y, this->z - p.z, this->w - p.w);
	}
	__device__ __host__ vector4d& vector4d::operator -= (const float i) {
		this->x -= i;
		this->y -= i;
		this->z -= i;
		this->w -= i;
		return *this;
	}
	__device__ __host__ vector4d& vector4d::operator -= (const vector4d p) {
		this->x -= p.x;
		this->y -= p.y;
		this->z -= p.z;
		this->w -= p.w;
		return *this;
	}

	__device__ __host__ vector4d  vector4d::operator * (const float i) const {
		return vector4d(
			this->x * i,
			this->y * i,
			this->z * i,
			this->w * i
		);
	}
	__device__ __host__ vector4d  vector4d::operator * (const vector4d p) const {
		return vector4d(
			this->x * p.x,
			this->y * p.y,
			this->z * p.z,
			this->w * p.w
		);
	}
	__device__ __host__ vector4d& vector4d::operator *= (const float i) {
		this->x *= i;
		this->y *= i;
		this->z *= i;
		this->w *= i;
		return *this;
	}
	__device__ __host__ vector4d& vector4d::operator *= (const vector4d p) {
		this->x *= p.x;
		this->y *= p.y;
		this->z *= p.z;
		this->w *= p.w;
		return *this;
	}

	__device__ __host__ vector4d  vector4d::operator / (const float i) const {
		return vector4d(
			this->x / i,
			this->y / i,
			this->z / i,
			this->w / i
		);
	}
	__device__ __host__ vector4d  vector4d::operator / (const vector4d p) const {
		return vector4d(
			this->x / p.x,
			this->y / p.y,
			this->z / p.z,
			this->w / p.w
		);
	}
	__device__ __host__ vector4d& vector4d::operator /= (const float i) {
		this->x /= i;
		this->y /= i;
		this->z /= i;
		this->w /= i;
		return *this;
	}
	__device__ __host__ vector4d& vector4d::operator /= (const vector4d p) {
		this->x /= p.x;
		this->y /= p.y;
		this->z /= p.z;
		this->w /= p.w;
		return *this;
	}












	// generic matrix implementation



	__device__ __host__ matrix4d::matrix4d() {
	}


	__device__ __host__ matrix4d::matrix4d(float i) {
		a11 = i;
		a22 = i;
		a33 = i;
		a44 = i;
		// ,
		//{	a41 + m.a41, a42 + m.a42, a43 + m.a43, a44 + m.a44	}
	}


	__device__ __host__ matrix4d::matrix4d(float data[4][4]) {
		a11 = data[0][0]; a12 = data[0][1]; a13 = data[0][2]; a14 = data[0][3];
		a21 = data[1][0]; a22 = data[1][1]; a23 = data[1][2]; a24 = data[1][3];
		a31 = data[2][0]; a32 = data[2][1]; a33 = data[2][2]; a34 = data[2][3];
		a41 = data[3][0]; a42 = data[3][1]; a43 = data[3][2]; a44 = data[3][3];
	}
	 


	__device__ __host__ matrix4d& matrix4d::operator = (const float i) {
		a11 = i; a12 = 0; a13 = 0; a14 = 0;
		a21 = 0; a22 = i; a23 = 0; a24 = 0;
		a31 = 0; a32 = 0; a33 = i; a34 = 0;
		a41 = 0; a42 = 0; a43 = 0; a44 = i;
		return *this;
	}


	__device__ __host__ matrix4d& matrix4d::operator = (const matrix4d& m) {
		a11 = m.a11; a12 = m.a12; a13 = m.a13; a14 = m.a14;
		a21 = m.a21; a22 = m.a22; a23 = m.a23; a24 = m.a24;
		a31 = m.a31; a32 = m.a32; a33 = m.a33; a34 = m.a34;
		a41 = m.a41; a42 = m.a42; a43 = m.a43; a44 = m.a44;
		return *this;						 
	}


	__device__ __host__ bool matrix4d::operator ==(const matrix4d& m) const {
		bool equal = true;
		equal &= a11 == m.a11 && a12 == m.a12 && a13 == m.a13 && a14 == m.a14;
		equal &= a21 == m.a21 && a22 == m.a22 && a23 == m.a23 && a24 == m.a24;
		equal &= a31 == m.a31 && a32 == m.a32 && a33 == m.a33 && a34 == m.a34;
		equal &= a41 == m.a41 && a42 == m.a42 && a43 == m.a43 && a44 == m.a44;
		return equal;

	}


	__device__ __host__ bool matrix4d::operator !=(const matrix4d& m) const {
		bool equal = false;
		equal |= a11 != m.a11 || a12 != m.a12 || a13 != m.a13 || a13 != m.a14;
		equal |= a21 != m.a21 || a22 != m.a22 || a23 != m.a23 || a23 != m.a24;
		equal |= a31 != m.a31 || a32 != m.a32 || a33 != m.a33 || a33 != m.a34;
		equal |= a41 != m.a41 || a42 != m.a42 || a43 != m.a43 || a44 != m.a44;
		return equal;

	}





	__device__ __host__ matrix4d  matrix4d::operator + (const matrix4d m) const {
		float temp[4][4] = {
				{	a11 + m.a11, a12 + m.a12, a13 + m.a13, a14 + m.a14	},
				{	a21 + m.a21, a22 + m.a22, a23 + m.a23, a24 + m.a24	},
				{	a31 + m.a31, a32 + m.a32, a33 + m.a33, a34 + m.a34	},
				{	a41 + m.a41, a42 + m.a42, a43 + m.a43, a44 + m.a44	}
		};
		return matrix4d(temp);
	}


	__device__ __host__ matrix4d& matrix4d::operator += (const matrix4d m) {
		a11 += m.a11; a12 += m.a12; a13 += m.a13; a14 += m.a14;
		a21 += m.a21; a22 += m.a22; a23 += m.a23; a24 += m.a24;
		a31 += m.a31; a32 += m.a32; a33 += m.a33; a34 += m.a34;
		a41 += m.a41; a42 += m.a42; a44 += m.a43; a43 += m.a44;
		return *this;

	}


	__device__ __host__ matrix4d  matrix4d::operator + (const float i) const {
		float temp[4][4] = {
				{	a11 + i, a12 + i, a13 + i, a14 + i	},
				{	a21 + i, a22 + i, a23 + i, a24 + i	},
				{	a31 + i, a32 + i, a33 + i, a34 + i	},
				{	a41 + i, a42 + i, a43 + i, a44 + i	}
		};
		return matrix4d(temp);

	}


	__device__ __host__ matrix4d& matrix4d::operator += (const float i) {
		a11 += i; a12 += i; a13 += i; a14 += i;
		a21 += i; a22 += i; a23 += i; a24 += i;
		a31 += i; a32 += i; a33 += i; a34 += i;
		a41 += i; a42 += i; a43 += i; a44 += i;
		return *this;
	}



	__device__ __host__ matrix4d  matrix4d::operator - (const matrix4d m) const {
		float temp[4][4] = {
				{	a11 - m.a11, a12 - m.a12, a13 - m.a13, a13 - m.a13	},
				{	a21 - m.a21, a22 - m.a22, a23 - m.a23, a23 - m.a23	},
				{	a31 - m.a31, a32 - m.a32, a33 - m.a33, a33 - m.a33	},
				{	a41 - m.a41, a42 - m.a42, a43 - m.a43, a44 - m.a44	}
		};
		return matrix4d(temp);
	}


	__device__ __host__ matrix4d& matrix4d::operator -= (const matrix4d m) {
		a11 -= m.a11; a12 -= m.a12;	 a13 -= m.a13; a14 -= m.a14;
		a21 -= m.a21; a22 -= m.a22;	 a23 -= m.a23; a24 -= m.a24;
		a31 -= m.a31; a32 -= m.a32;	 a33 -= m.a33; a34 -= m.a34;
		a41 -= m.a41; a42 -= m.a42;	 a44 -= m.a44; a44 -= m.a44;
		return *this;
	}


	__device__ __host__ matrix4d  matrix4d::operator - (const float i) const {
		float temp[4][4] = {
				{	a11 - i, a12 - i, a13 - i, a14 - i	},
				{	a21 - i, a22 - i, a23 - i, a24 - i	},
				{	a31 - i, a32 - i, a33 - i, a34 - i	},
				{	a41 - i, a42 - i, a43 - i, a44 - i	}
		};
		return matrix4d(temp);
	}


	__device__ __host__ matrix4d& matrix4d::operator -= (const float i) {
		a11 -= i; a12 -= i; a13 -= i; a14 -= i;
		a21 -= i; a22 -= i; a23 -= i; a24 -= i;
		a31 -= i; a32 -= i; a33 -= i; a34 -= i;
		a41 -= i; a42 -= i; a43 -= i; a44 -= i;
		return *this;
	}



	__device__ __host__ matrix4d  matrix4d::operator * (const matrix4d m) const {
		float temp[4][4] = {
				{	// first row
					a11 * m.a11 + a12 * m.a21 + a13 * m.a31  +a14 * m.a41,
					a11 * m.a12 + a12 * m.a22 + a13 * m.a32  +a14 * m.a42,
					a11 * m.a13 + a12 * m.a23 + a13 * m.a33  +a14 * m.a43,
					a11 * m.a14 + a12 * m.a24 + a13 * m.a34 + a14 * m.a44,
				},
				{	// second row		     
					a21 * m.a11 + a22 * m.a21 + a23 * m.a31 + a24 * m.a41,
					a21 * m.a12 + a22 * m.a22 + a23 * m.a32 + a24 * m.a42,
					a21 * m.a13 + a22 * m.a23 + a23 * m.a33 + a24 * m.a43,
					a21 * m.a14 + a22 * m.a24 + a23 * m.a34 + a24 * m.a44,
				},
				{	// third row		     
					a31 * m.a11 + a32 * m.a21 + a33 * m.a31 + a34 * m.a41,
					a31 * m.a12 + a32 * m.a22 + a33 * m.a32 + a34 * m.a42,
					a31 * m.a13 + a32 * m.a23 + a33 * m.a33 + a34 * m.a43,
					a31 * m.a14 + a32 * m.a24 + a33 * m.a34 + a34 * m.a44,
				},
				{	// fourth row		     
					a41 * m.a11 + a42 * m.a21 + a43 * m.a31 + a44 * m.a41,
					a41 * m.a12 + a42 * m.a22 + a43 * m.a32 + a44 * m.a42,
					a41 * m.a13 + a42 * m.a23 + a43 * m.a33 + a44 * m.a43,
					a41 * m.a14 + a42 * m.a24 + a43 * m.a34 + a44 * m.a44,
				}
		};
		return matrix4d(temp);
	}


	__device__ __host__ matrix4d& matrix4d::operator *= (const matrix4d m) {
		// first row
		a11 = a11 * m.a11 + a12 * m.a21 + a13 * m.a31 + a14 * m.a41;
		a12 = a11 * m.a12 + a12 * m.a22 + a13 * m.a32 + a14 * m.a42;
		a13 = a11 * m.a13 + a12 * m.a23 + a13 * m.a33 + a14 * m.a43;
		a14 = a11 * m.a14 + a12 * m.a24 + a13 * m.a34 + a14 * m.a44;

		// second row		     
		a21 = a21 * m.a11 + a22 * m.a21 + a23 * m.a31 + a24 * m.a41;
		a22 = a21 * m.a12 + a22 * m.a22 + a23 * m.a32 + a24 * m.a42;
		a23 = a21 * m.a13 + a22 * m.a23 + a23 * m.a33 + a24 * m.a43;
		a24 = a21 * m.a14 + a22 * m.a24 + a23 * m.a34 + a24 * m.a44;

		// third row		     
		a31 = a31 * m.a11 + a32 * m.a21 + a33 * m.a31 + a34 * m.a41;
		a32 = a31 * m.a12 + a32 * m.a22 + a33 * m.a32 + a34 * m.a42;
		a33 = a31 * m.a13 + a32 * m.a23 + a33 * m.a33 + a34 * m.a43;
		a34 = a31 * m.a14 + a32 * m.a24 + a33 * m.a34 + a34 * m.a44;

		// third row		     
		a41 = a41 * m.a11 + a42 * m.a21 + a43 * m.a31 + a44 * m.a41;
		a42 = a41 * m.a12 + a42 * m.a22 + a43 * m.a32 + a44 * m.a42;
		a43 = a41 * m.a13 + a42 * m.a23 + a43 * m.a33 + a44 * m.a43;
		a44 = a41 * m.a14 + a42 * m.a24 + a43 * m.a34 + a44 * m.a44;

		return *this;
	}


	__device__ __host__ matrix4d  matrix4d::operator * (const float i) const {
		float temp[4][4] = {
				{	a11 * i, a12 * i, a13 * i, a14 * i	},
				{	a21 * i, a22 * i, a23 * i, a24 * i	},
				{	a31 * i, a32 * i, a33 * i, a34 * i	},
				{	a41 * i, a42 * i, a43 * i, a44 * i	}
		};
		return matrix4d(temp);
	}


	__device__ __host__ matrix4d& matrix4d::operator *= (const float i) {
		a11 *= i; a12 *= i; a13 *= i; a14 *= i;
		a21 *= i; a22 *= i; a23 *= i; a24 *= i;
		a31 *= i; a32 *= i; a33 *= i; a34 *= i;
		a41 *= i; a42 *= i; a43 *= i; a44 *= i;
		return *this;

	}



	__device__ __host__ matrix4d  matrix4d::operator / (const float i) const {
		float temp[4][4] = {
				{	a11 / i, a12 / i, a13 / i, a14 / i	},
				{	a21 / i, a22 / i, a23 / i, a24 / i	},
				{	a31 / i, a32 / i, a33 / i, a34 / i	},
				{	a41 / i, a42 / i, a43 / i, a44 / i	}
		};
		return matrix4d(temp);
	}


	__device__ __host__ matrix4d& matrix4d::operator /= (const float i) {
		a11 /= i; a12 /= i;	  a13 /= i; a14 /= i;
		a21 /= i; a22 /= i;	  a23 /= i; a24 /= i;
		a31 /= i; a32 /= i;	  a33 /= i; a34 /= i;
		a41 /= i; a42 /= i;	  a43 /= i; a44 /= i;
		return *this;
	}




	__device__ __host__ vector4d  matrix4d::operator * (const vector4d v) const {
		float x = a11 * v.x + a12 * v.y + a13 * v.z + a14 * v.w;
		float y = a21 * v.x + a22 * v.y + a23 * v.z + a24 * v.w;
		float z = a31 * v.x + a32 * v.y + a33 * v.z + a34 * v.w;
		float w = a41 * v.x + a42 * v.y + a43 * v.z + a44 * v.w;
		vector4d temp = vector4d(x, y, z, w);
		return temp;
	}















}