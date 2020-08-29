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
	std::string file = "trim_easy";
	//std::string file = "trim_medium";
	//std::string file = "simpleHill4";
    cv::Mat image ;
    image = cv::imread("data/"+file+".tif",cv::IMREAD_UNCHANGED);
	if(!image.data)
	{
		std::cout << "Error: Could not open or locate the file." << std::endl;
        return -1;
	}
    //Watershed w(image);
    //w.applyWatershed();
	Segmentation s(image,file);
	s.segment();
	return 0;
}
