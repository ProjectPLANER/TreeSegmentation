#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "json.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include "Image.h"

namespace json = nlohmann;

Image::Image() {}
Image::~Image() {}
Image::Image(std::string dem) {}

Image::Image(std::string dem, std::string mask)
{
    this->dem = cv::imread(dem,cv::IMREAD_UNCHANGED);
    std::ifstream ifs(mask);
    json::json j;
    ifs >> j;
    for(size_t i = 0; i < j["features"][0]["geometry"]["coordinates"][0].size(); ++i)
    {
        cv::Point2d p(j["features"][0]["geometry"]["coordinates"][0][i][0],j["features"][0]["geometry"]["coordinates"][0][i][1]);
        maskPoints.push_back(p);
    }
    //std::cout << j["features"][0]["geometry"]["coordinates"][0].size() << std::endl;
    for(size_t i = 0; i < j["features"][0]["geometry"]["coordinates"][0].size(); ++i)
    {
        std::cout << maskPoints[i].x << " " << maskPoints[i].y << std::endl;
    }
}

void Image::applyMask(std::string s)
{
    
}