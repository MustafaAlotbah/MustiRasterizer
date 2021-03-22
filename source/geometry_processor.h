#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"algebra.h"
#include <vector>


namespace mge {

	struct mesh2d {
		std::vector<vector2d> vectors;
	};

	struct mesh {
		std::vector<vector4d> vectors;
	};



	mesh2d translateScale(mesh2d mesh, float scale, vector2d offset);








}