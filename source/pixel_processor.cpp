#include"pixel_processor.h"
#include<cmath>

/*
 Pixel Processing
1-  Pixel Shading
2-	Merging


*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"



namespace mge
{

	/*-----------------PIXELS------------------*/

	__host__ __device__ Pixel::Pixel(uint32_t value, float depth) {
		color.value = value;
		this->depth = depth;
	}



	__host__ __device__ Pixel::Pixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha, float depth) :
		Pixel(red | (green << 8) | (blue << 16) | (alpha << 24), depth)
	{
	}

	__host__  Pixel::Pixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha) :
		Pixel(red, green, blue, alpha, 0) {
	}


	__host__ __device__ Pixel::Pixel(uint32_t value) :
		Pixel(value, depth)
	{
	}

	Pixel::Pixel() {
		color.rgba.r = 0;
		color.rgba.g = 0;
		color.rgba.b = 0;
		color.rgba.a = 255;
		this->depth = 0;
	}

	bool Pixel::operator==(const Pixel& p) const {
		return color.value == p.color.value;
	}


	bool Pixel:: operator !=(const Pixel& p) const {
		return color.value != p.color.value;
	}

	Pixel Pixel::operator *(const float i) const {
		float r = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.r) * i)));
		float g = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.g) * i)));
		float b = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.b) * i)));
		return Pixel(r, g, b, color.rgba.a);
	}


	Pixel Pixel::operator /(const float i) const {
		float r = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.r) / i)));
		float g = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.g) / i)));
		float b = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.b) / i)));
		return Pixel(r, g, b, color.rgba.a);
	}


	Pixel& Pixel::operator *=(const float i) {
		this->color.rgba.r = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.r) * i)));
		this->color.rgba.g = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.g) * i)));
		this->color.rgba.b = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.b) * i)));
		return *this;
	}


	Pixel& Pixel::operator /=(const float i) {
		this->color.rgba.r = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.r) / i)));
		this->color.rgba.g = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.g) / i)));
		this->color.rgba.b = uint8_t(fmin(255.0f, fmax(0.0f, float(this->color.rgba.b) / i)));
		return *this;
	}
	Pixel Pixel::operator +(const Pixel& p)const {
		float r = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.r + p.color.rgba.r)));
		float g = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.g + p.color.rgba.g)));
		float b = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.b + p.color.rgba.b)));
		return Pixel(r, g, b, color.rgba.a);
	}
	Pixel Pixel::operator -(const Pixel& p) const {
		float r = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.r - p.color.rgba.r)));
		float g = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.g - p.color.rgba.g)));
		float b = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.b - p.color.rgba.b)));
		return Pixel(r, g, b, color.rgba.a);
	}
	Pixel& Pixel::operator +=(const Pixel& p) {
		this->color.rgba.r = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.r + p.color.rgba.r)));
		this->color.rgba.g = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.g + p.color.rgba.g)));
		this->color.rgba.b = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.b + p.color.rgba.b)));
		return *this;
	}
	Pixel& Pixel::operator -=(const Pixel& p) {
		this->color.rgba.r = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.r - p.color.rgba.r)));
		this->color.rgba.g = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.g - p.color.rgba.g)));
		this->color.rgba.b = uint8_t(fmin(255.0f, fmax(0.0f, this->color.rgba.b - p.color.rgba.b)));
		return *this;
	}
	Pixel Pixel::inv()const {
		float r = uint8_t(fmin(255.0f, fmax(0.0f, 255 - this->color.rgba.r )));
		float g = uint8_t(fmin(255.0f, fmax(0.0f, 255 - this->color.rgba.g )));
		float b = uint8_t(fmin(255.0f, fmax(0.0f, 255 - this->color.rgba.b )));
		return Pixel(r, g, b, color.rgba.a);
	}





	Pixel PixelFloat(float red, float green, float blue, float alpha) {
		return Pixel(uint8_t(red * 255.0f), uint8_t(green * 255.0f), uint8_t(blue * 255.0f), uint8_t(alpha * 255.0f));
	}
	Pixel PixelLerp(const Pixel& p1, const Pixel& p2, float t) {
		return (p2 * t) + p1 * (1.0f - t);
	}



	/*-----------------SPRITES------------------*/



}