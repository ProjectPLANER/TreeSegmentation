#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>
#include <set>
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
	//std::string file = "noisy";

	std::string maps[34];
	maps[0] = "contourHill";
	maps[1] = "contourHillEasy";
	maps[2] = "contourHillJoin";
	maps[3] = "contourHillJoinEasy";
	maps[4] = "contourHillJoinSpread";
	maps[5] = "contourHillSpread";
	maps[6] = "flat";
	maps[7] = "flatEasy";
	maps[8] = "flatSpread";
	maps[9] = "flatSpreadEasy";
	maps[10] = "gentle";
	maps[11] = "gentleEasy";
	maps[12] = "gentleSpread";
	maps[13] = "hills";
	maps[14] = "hillsEasy";
	maps[15] = "hillsSmooth";
	maps[16] = "hillsSpread";
	maps[17] = "steep";
	maps[18] = "steepEasy";
	maps[19] = "steepSpread";

	maps[20] = "contourHillJoinSmall";
	maps[21] = "contourHillJoinSmallEasy";
	maps[22] = "contourHillSmall";
	maps[23] = "contourHillSmallEasy";
	maps[24] = "contourHillSmallSpread";
	maps[25] = "flatSmall";
	maps[26] = "flatSmallEasy";
	maps[27] = "flatSmallSpread";
	maps[28] = "gentleSmall";
	maps[29] = "gentleSmallEasy";
	maps[30] = "hillsSmall";
	maps[31] = "hillsSmallEasy";
	maps[32] = "steepSmall";
	maps[33] = "steepSmallEasy";
	
    cv::Mat image;
	for (size_t i = 0; i < 20; i++)
	{
		std::cout << maps[i] << std::endl;
		image = cv::imread("data/"+maps[i]+"/final.tif",cv::IMREAD_UNCHANGED);
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
	
	/*std::cout << file << std::endl;
    image = cv::imread("data/"+file+".tif",cv::IMREAD_UNCHANGED);
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
