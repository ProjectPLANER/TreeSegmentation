#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "Watershed.h"

int main(int, char**) 
{
    cv::Mat image ;
    //image = cv::imread("data/dem_full_easy.tif",cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
    image = cv::imread("data/dem_full_easy.tif",cv::IMREAD_UNCHANGED);
	if(!image.data)
	{
		std::cout << "Error: Could not open or locate the file." << std::endl;
        return -1;
	}
	//cv::namedWindow("Disp",cv::WINDOW_AUTOSIZE);
	//cv::imshow("Disp",*image);
	//cv::waitKey(0);
    Watershed w(image);
    w.applyWatershed();
	return 0;
}
