

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
	}
	__device__ __host__ matrix2d& matrix2d::operator = (const matrix2d& p) {
		this->a11 = p.a11;
		this->a12 = p.a12;
		this->a21 = p.a21;
		this->a22 = p.a22;
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


}