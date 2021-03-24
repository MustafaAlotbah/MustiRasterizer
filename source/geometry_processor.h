#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"algebra.h"
#include <vector>
#include <fstream>
#include <strstream>
#include "dyn_algebra.h"



namespace mge {

	struct path2d {
		std::vector<vector2d> vectors;

	};


	struct path4d {
		std::vector<vector4d> vectors;

		__device__ __host__ path4d(vector4d a, vector4d b, vector4d c);
	};

	// here the path has only 3 points!
	struct Mesh {
		std::vector<path4d> triags;

		bool loadFromFile(std::string fileName);

	};


	struct dyn_path4d {
		std::vector<cvector4d> vectors;

		__device__ __host__ dyn_path4d(cvector4d a, cvector4d b, cvector4d c);
	};

	// here the path has only 3 points!
	struct dyn_Mesh {
		std::vector<dyn_path4d> triags;

		bool loadFromFile(std::string fileName);

	};



	path2d translateScale(path2d mesh, float scale, vector2d offset);

























}