

/*
Rasterization
1- triangle setup
2- triangle traversal


This Rasterizer works on CPU
*/


#include"rasterizer.h"






namespace mge {










	Rasterizer::Rasterizer(VideoBuffer* buffer) {
		this->buffer = buffer;
	}
	Rasterizer::~Rasterizer() {

	}



	/*
	Pixel has its x, y, z and depth information
	Note the the pixel has to be in the screen space already!
	*/
	bool Rasterizer::drawPixel(int x, int y, Pixel p) {
		y = buffer->height - y -1;	// correct the orientation
		*((uint32_t*)buffer->addr + y * buffer->width + x) = p.color.value;
		return true;
	}



	// This line looks like this ( | )
	bool Rasterizer::drawVerticalLine(int x, int y1, int y2, Pixel p) {
		int startY = min(min(y1, y2), buffer->height);
		int endY = max(max(y1, y2), 0);
		for (int y = startY; y <= endY; y++)
		{
			drawPixel(x, y, p);
		}
		return true;
	}


	// This line looks like this ( - )
	bool Rasterizer::drawHorizontalLine(int y, int x1, int x2, Pixel p) {
		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);
		for (int x = startX; x <= endX; x++)
		{
			drawPixel(x, y, p);
		}
		return true;
	}


	// this line looks like ( \ )
	bool Rasterizer::drawFallRightLine(int x1, int y1, int x2, int y2, Pixel p) {

		int startX = min(min(x1, x2), buffer->width);
		int endX = max(max(x1, x2), 0);

		int startY = min(min(y1, y2), buffer->height);
		int endY = max(max(y1, y2), 0);

		int currY = startY;
		int prevY = startY;
		for (int x = startX; x <= endX; x++)
		{
			currY = startY + (x - startX) * (float(endY - startY) / float(endX - startX));
			drawVerticalLine(x, prevY, currY, p);
			prevY = currY;
		}

		return true;
	}


	// this line looks like ( / )
	bool Rasterizer::drawFallLeftLine(int x1, int y1, int x2, int y2, Pixel p) {

		int xMin = min(min(x1, x2), buffer->width);
		int xMax = max(max(x1, x2), 0);

		int yMax = max(max(y1, y2), 0);
		int yMin = min(min(y1, y2), buffer->height);

		int currY = yMax;
		int prevY = yMax;
		int f;
		for (int x = xMin; x <= xMax; x++)
		{
			currY = yMax - (x - xMin) * (float(yMax - yMin) / float(xMax - xMin));
			drawVerticalLine(x, prevY, currY, p);
			prevY = currY;
		}
		return true;
	}



	bool Rasterizer::drawLine(int x1, int y1, int x2, int y2, Pixel p) {

		if (x1 == x2)
		{
			return drawVerticalLine(x1, y1, y2, p);
		}
		else if (y1 == y2)
		{
			return drawHorizontalLine(y1, x1, x2, p);
		}
		else 
		{
			int dx = x2 - x1;
			int dy = y2 - y1;

			if (dx > 0 && dy > 0)
			{
				return drawFallRightLine(x1, y1, x2, y2, p);
			}
			else if (dx < 0 && dy < 0) {
				return drawFallRightLine(x2, y2, x1, y1, p);
			}
			else if (dx > 0 && dy < 0) {
				return drawFallLeftLine(x1, y1, x2, y2, p);
			}
			else {
				return drawFallLeftLine(x2, y2, x1, y1, p);
			}

		}




		return true;
	}








}