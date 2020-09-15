#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>
#include "Segmentation.h"
#include "Image.h"
#include "Watershed.h"

int main(int, char**) 
{
	//std::string file = "trim_easy";
	//std::string file = "trim_medium";
	//std::string file = "trim_hard";
	//std::string file = "simpleHill4";
	//std::string file = "easyHill";
	//std::string file = "flat";
	//std::string file = "steps";
	//std::string file = "bigTrees";
	//std::string file = "sameHeights";
	//std::string file = "spacedTrees";
	//std::string file = "touching2";
	//std::string file = "circleTree2";
	//std::string file = "flatOverlapRounded";
	//std::string file = "1";
	//std::string file = "2";
	//std::string file = "3";
	//std::string file = "hillsGradual";
	//std::string file = "hillsSteep";
	//std::string file = "valleys";
	//std::string file = "contoursSteep";
	std::string file = "noisy";

	std::string maps[11];
	maps[0] = "flatOverlap/final";
	maps[1] = "flatOverlapSimple/final";
	maps[2] = "flatSmallOverlap/final";
	maps[3] = "flatSmallOverlapSimple/final";
	maps[4] = "flatSmallSpread/final";
	maps[5] = "flatSmallSpreadSimple/final";
	maps[6] = "flatSpread/final";
	maps[7] = "flatSpreadSimple/final";
	maps[8] = "hillsOverlap/final";
	maps[9] = "hillsSmoothOverlap/final";
	maps[10] = "hillsSpread/final";
    cv::Mat image;
	for (size_t i = 6; i < 11; i++)
	{
		std::cout << maps[i] << std::endl;
		image = cv::imread("data/"+maps[i]+".tif",cv::IMREAD_UNCHANGED);
		if(!image.data)
		{
			std::cout << "Error: Could not open or locate the file." << std::endl;
        	return -1;
		}
    //Watershed w(image);
    //w.applyWatershed();
	Segmentation s(image,maps[i]);
	s.segment();
	}
	
    /*image = cv::imread("data/"+file+".tif",cv::IMREAD_UNCHANGED);
	if(!image.data)
	{
		std::cout << "Error: Could not open or locate the file." << std::endl;
        return -1;
	}
    //Watershed w(image);
    //w.applyWatershed();
	Segmentation s(image,file);
	s.segment();*/
	return 0;
}
