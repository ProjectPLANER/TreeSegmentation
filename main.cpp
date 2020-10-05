#include <string>
#include <set>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Segmentation.h"

int main(int argc, char* argv[]) 
{
	std::string maps[37];
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

	maps[34] = "trim_easy";
	maps[35] = "trim_medium";
	maps[36] = "trim_hard";
	
    /*cv::Mat image;
	for (size_t i = 8; i < 9; i++)
	{
		std::cout << maps[i] << std::endl;
		image = cv::imread("../data/"+maps[i]+"/final.tif",cv::IMREAD_UNCHANGED);
		if(!image.data)
		{
			std::cout << "Error: Could not open or locate the file." << std::endl;
        	return -1;
		}

	Segmentation s(image,maps[i],0,28,25);
	s.segment();
	}*/
	
	if(argc < 5 || argc > 6)
	{
		std::cout << "Please enter the correct arguments" << std::endl;
		std::cout << "Format: -t(optional) <fileNumber> <minimumPixelValue> <SLICSize> <localMaxWindowSize>" << std::endl;
		return -1; 
	}

	if(argc == 5)
	{
		std::cout << maps[std::atoi(argv[1])] << std::endl;
		cv::Mat image = cv::imread("../data/"+maps[std::atoi(argv[1])]+"/final.tif",cv::IMREAD_UNCHANGED);
		if(!image.data)
		{
			std::cout << "Error: Could not open or locate the file." << std::endl;
        	return -1;
		}
		Segmentation s(image, maps[std::atoi(argv[1])], std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
		s.segment(false);
		return 0;
	}

	if(argc == 6)
	{
		if(std::string(argv[1]) != "-t")
		{
			std::cout << "Please enter the correct arguments" << std::endl;
			std::cout << "Format: -t(optional) <fileNumber> <minimumPixelValue> <SLICSize> <localMaxWindowSize>" << std::endl;
			return -1; 
		}

		std::cout << maps[std::atoi(argv[2])] << std::endl;
		cv::Mat image = cv::imread("../data/"+maps[std::atoi(argv[2])]+"/final.tif",cv::IMREAD_UNCHANGED);
		if(!image.data)
		{
			std::cout << "Error: Could not open or locate the file." << std::endl;
        	return -1;
		}
		Segmentation s(image, maps[std::atoi(argv[2])], std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
		s.segment(true);
		return 0;
	}
}