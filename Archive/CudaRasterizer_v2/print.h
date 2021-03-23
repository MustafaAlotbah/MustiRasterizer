#pragma once
#include "algebra.h"
#include"geometry_processor.h"
#include <vector>

#define CHAR_HALF_WIDTH 0.5
#define CHAR_HALF_HEIGHT 1



namespace mge {
	

	path2d zero = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT * 0.7),
			vector2d(0, CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT * 0.7),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT * 0.7),
			vector2d(0, -CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT * 0.7),
			vector2d(-CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT * 0.7),
		}
	};


	path2d one = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT * 0.7),
			vector2d(0, -CHAR_HALF_HEIGHT),
			vector2d(0, CHAR_HALF_HEIGHT)
		}
	};

	path2d two = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT * 0.7),
			vector2d(0, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT * 0.7),
			vector2d(-CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT)
		}
	};

	path2d three = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(0, 0),
			vector2d(CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
		}
	};

	path2d four = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, 0),
			vector2d(CHAR_HALF_WIDTH, 0),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
		}
	};
	path2d five = {
		std::vector<vector2d>{
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT* 0.5),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT * 0.5),
			vector2d(CHAR_HALF_WIDTH,  CHAR_HALF_HEIGHT * 0.5),
			vector2d(-CHAR_HALF_WIDTH,  CHAR_HALF_HEIGHT),
		}
	};

	path2d six = {
		std::vector<vector2d>{
			vector2d(0, -CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH,  CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH,  CHAR_HALF_HEIGHT),
			vector2d(0,  0),
		}
	};
	path2d seven = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(0,  CHAR_HALF_HEIGHT),
		}
	};
	path2d eight = {
		std::vector<vector2d>{
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
		}
	};
	path2d nine = {
		std::vector<vector2d>{
			vector2d(CHAR_HALF_WIDTH, 0),
			vector2d(-CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, -CHAR_HALF_HEIGHT),
			vector2d(CHAR_HALF_WIDTH, CHAR_HALF_HEIGHT),
		}
	};

	



	std::vector<path2d> chars = {zero, one, two, three, four, five, six, seven, eight, nine};

	uint32_t x;


}