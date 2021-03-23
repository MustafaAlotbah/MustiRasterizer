#pragma once
/*
This should deal with pixel manipulations

*/

#include"types.h"
#include<vector>

namespace mge
{

	class Pixel
	{
		union Color	{
			uint32_t value = 0;
			struct rgba
			{
				uint8_t r; uint8_t g; uint8_t b; uint8_t a;
			} rgba;
		};

	public:
		uint16_t depth = 0;
		Color color;
	
		enum Mode
		{
			NORMAL, MASK, ALPHA
		};

		Pixel();
		Pixel(uint32_t value, uint16_t depth);
		Pixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha);
		Pixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha, uint16_t depth);
		Pixel(uint32_t value);

		Pixel& operator =(const Pixel& v) = default;
		bool operator ==(const Pixel& p) const;
		bool operator !=(const Pixel& p) const;
		Pixel operator *(const float i) const;
		Pixel operator /(const float i) const;
		Pixel& operator *=(const float i) ;
		Pixel& operator /=(const float i) ;
		Pixel operator +(const Pixel& p) const;
		Pixel operator -(const Pixel& p) const;
		Pixel& operator +=(const Pixel& p) ;
		Pixel& operator -=(const Pixel& p) ;
		Pixel inv() const;
	};






	Pixel PixelFloat(float red, float green, float blue, float alpha = 1.0f);
	Pixel PixelLerp(const Pixel& p1, const Pixel& p2, float t);





}