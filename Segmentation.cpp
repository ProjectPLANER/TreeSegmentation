#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include <set>
#include "Preprocessing.h"
#include "Watershed.h"
#include "Segmentation.h"
#include "Evaluation.h"

#include <iostream>


Segmentation::Segmentation(cv::Mat& image, std::string file) 
{
    this->image = image;
    this->file = file;
}

Segmentation::~Segmentation() {}

void Segmentation::segment()
{   
    Preprocessing p;

    //Create a mask image of the DEM
    cv::Mat mask = p.applyThreshold(image,0,1); //easy: (image,0,1); medium:(image,-2,1); hard:(image,19,1)
    mask.convertTo(mask,CV_8U);
    std::cout << "Mask image created..." << std::endl;

    //If the DEM is not rectangular, set the null value to 0
    p.removeBoundary(image,0); //16 for easy
    std::cout << "Boundary removed..." << std::endl;

    //Calculate the local maxima in the DEM
    cv::Mat localMax = p.findLocalMax(image);
    for (size_t i = 0; i < mask.rows; i++)
    {
        for (size_t j = 0; j < mask.cols; j++)
        {
            if (mask.at<uint8_t>(i,j) == 0)
            {
                localMax.at<uint8_t>(i,j) = 0;
            }
            if (i == 0 || i == mask.rows-1 || j == 0 || j == mask.cols-1)
            {
                localMax.at<uint8_t>(i,j) = 0;
            }
        }
    }
    std::cout << "Local maxima calculated..." << std::endl;

    cv::Mat image_8U;
    cv::normalize(image, image_8U, 0, 255, cv::NORM_MINMAX);
    image_8U.convertTo(image_8U,CV_8U);

    //test
    /*cv::Mat gt = cv::imread("data/groundTruth_easy.tif",cv::IMREAD_UNCHANGED);
    for (size_t i = 0; i < gt.rows; i++)
    {
        for (size_t j = 0; j < gt.cols; j++)
        {
            //cv::Vec4b pix = gt.at<cv::Vec4b>(i,j);
            if(gt.at<uint8_t>(i,j) == 255)
            {
                image_8U.at<uint8_t>(i,j) = 255;
            }
        }
    }*/
    

    //endtest
    //cv::imwrite(file+"_8UC1.tif",image_8U);
    image_8U = p.applyFilterGausian(image_8U);
    cv::imwrite(file+"_8UC1.tif",image_8U);

    cv::Mat slicmask, result;
    int min_element_size = 25;
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(image,cv::ximgproc::SLICO,28,0.95); //30,10,   28
    slic->iterate(10);
    slic->getLabelContourMask(slicmask, true);
    cv::Mat clusters = cv::Mat::zeros(image_8U.size(),CV_32S);
    std::cout << slic->getNumberOfSuperpixels() << " superpixel clusters found..." << std::endl;
    slic->getLabels(clusters);

    std::set<int> isTree;
    std::vector<uint8_t> mapping[slic->getNumberOfSuperpixels()];
    std::vector<cv::Point> coord[slic->getNumberOfSuperpixels()];

    for (size_t i = 0; i < localMax.rows; i++)
    {
        for (size_t j = 0; j < localMax.cols; j++)
        {
            mapping[clusters.at<int32_t>(i,j)].push_back(image_8U.at<uint8_t>(i,j));
            coord[clusters.at<int32_t>(i,j)].push_back(cv::Point(i,j));
            if (localMax.at<uint8_t>(i,j) == 255)
            {
                isTree.insert(clusters.at<int32_t>(i,j));
            }
        }
    }

    //Find the neighbour of each superpixel
    std::vector<int> neighbours[slic->getNumberOfSuperpixels()];
    p.findNeighbours(clusters,neighbours);

    //Find the average variance of tree superpixels
    float avgVar = 0;
    int numTrees = 0;
    for (int i = 0; i < slic->getNumberOfSuperpixels(); i++)
    {
        if (isTree.find(i) != isTree.end())
        {
            avgVar += p.variance(mapping[i]);
            numTrees++;        
        }
    }
    avgVar = avgVar/numTrees;
    float thresVar = avgVar - (avgVar*0.9);

    for (int i = 0; i < slic->getNumberOfSuperpixels(); i++)
    {
        if (isTree.find(i) == isTree.end())
        {
            int count = 0;
            for (int j = 0; j < neighbours[i].size(); j++)
            {
                if (isTree.find(neighbours[i][j]) != isTree.end())
                {
                    count++;
                }               
            }
            
            if ((count == 1 || count == 2 || count == 3) && p.variance(mapping[i]) > thresVar)
            {
                std::cout << "ming" << std::endl;
                cv::Point maxPos = p.getMaxPosition(mapping[i],coord[i]);
                localMax.at<uint8_t>(maxPos.x,maxPos.y) = 255;
                isTree.insert(i); 
            }                 
        }
    }

    cv::Mat image_8UC3;
    cv::cvtColor(image_8U,image_8UC3,cv::COLOR_GRAY2RGB);
    std::cout << "DEM cleaned-up..." << std::endl;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(localMax, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
     // Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(localMax.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    std::cout << "Contours calculated..." << std::endl;

    cv::Mat treeSLICMask = cv::Mat::zeros(image_8U.size(),CV_8U);
    for (size_t i = 0; i < treeSLICMask.rows; i++)
    {
        for (size_t j = 0; j < treeSLICMask.cols; j++)
        {
            if(isTree.find(clusters.at<int32_t>(i,j)) != isTree.end())
            {
                treeSLICMask.at<uint8_t>(i,j) = 255;
            }
        }
    }
    cv::imwrite("treeSlicMask.tif",treeSLICMask);
    cv::Mat edges = p.findEdges(image_8U);
    cv::imwrite("edges.tif",edges);


    cv::Mat newImg = image_8UC3.clone();
    for (size_t i = 0; i < newImg.rows; i++)
    {
        for (size_t j = 0; j < newImg.cols; j++)
        {
            if (treeSLICMask.at<uint8_t>(i,j) == 0)
            {
                newImg.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            } 
        }
    }
    
    

    //Region grow
/*
    cv::Mat regionGrow;
    cv::cvtColor(newImg,regionGrow,cv::COLOR_RGB2GRAY);
    for (size_t i = 0; i < localMax.rows; i++)
    {
        for (size_t j = 0; j < localMax.cols; j++)
        {
            if(localMax.at<uint8_t>(i,j) == 255 && regionGrow.at<uint8_t>(i,j) != 0)
            {
                p.regionGrow(regionGrow,i,j,255);
            }       
        }
    }
    cv::imwrite("regionGrow.tif",regionGrow);
    cv::Mat regionGrowColour = cv::Mat::zeros(image_8U.size(),CV_8UC4);
    //cv::cvtColor(newImg,regionGrowColour,cv::COLOR_RGB2RGBA);
    for (size_t i = 0; i < localMax.rows; i++)
    {
        for (size_t j = 0; j < localMax.cols; j++)
        {
            if(regionGrow.at<uint8_t>(i,j) == 255)
            {
                regionGrowColour.at<cv::Vec4b>(i,j) = cv::Vec4b(0,0,255,100);
            }       
        }
    }
    cv::imwrite("regionGrowColour.tif",regionGrowColour);

*/
    //End region grow
    

    cv::imwrite("markers.tif",markers);
    cv::imwrite("newImg.tif",newImg);
    cv::watershed(newImg,markers); //image_8UC3
    std::cout << "Watershed complete..." << std::endl;
    cv::imwrite("output/"+file+"_watershedLines.tif",markers);

    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
        //imshow("Markers_v2", mark);
    // image looks like at that point
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    cv::imwrite("output/"+file+"_watershedBasins.tif",dst);
    // Visualize the final image
    display(image_8UC3,dst,slicmask,localMax);

    //evaluate(treeSLICMask,isTree,coord);
    //cv::Mat watershedMask = getWatershedMask(mark);
    //evaluate(watershedMask);
    //evaluate(markers,treeSLICMask,localMax,contours.size());
    evaluatePer(markers,contours.size());

    cv::namedWindow("Final Result",cv::WINDOW_NORMAL);
    cv::resizeWindow("Final Result", 1920,1080);
    cv::imshow("Final Result", dst);
    //cv::waitKey(0);
}

cv::Mat Segmentation::getWatershedMask(cv::Mat& image)
{
    cv::Mat watershedMask = cv::Mat::zeros(image.size(),CV_8U);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uint8_t>(i,j) > 0)
                watershedMask.at<uint8_t>(i,j) = 255;
        }
    }
    cv::imwrite("ttttttttttest.tif",watershedMask);
    return watershedMask;
}

void Segmentation::display(cv::Mat& image,cv::Mat& boundaries,cv::Mat& slicBoundaries,cv::Mat& localMax)
{
    cv::Mat newImg = image.clone();
    cv::cvtColor(newImg,newImg,cv::COLOR_RGB2RGBA);
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {           
            newImg.at<cv::Vec4b>(i,j)[0] = (uint8_t)(boundaries.at<cv::Vec3b>(i,j)[0] * (newImg.at<cv::Vec4b>(i,j)[0] / 255.0));
            newImg.at<cv::Vec4b>(i,j)[1] = (uint8_t)(boundaries.at<cv::Vec3b>(i,j)[1] * (newImg.at<cv::Vec4b>(i,j)[1] / 255.0));
            newImg.at<cv::Vec4b>(i,j)[2] = (uint8_t)(boundaries.at<cv::Vec3b>(i,j)[2] * (newImg.at<cv::Vec4b>(i,j)[2] / 255.0));
            newImg.at<cv::Vec4b>(i,j)[3] = 255;
            if (slicBoundaries.at<uint8_t>(i,j) != 0)
            {
                newImg.at<cv::Vec4b>(i,j)[0] = 0;
                newImg.at<cv::Vec4b>(i,j)[1] = 0;
                newImg.at<cv::Vec4b>(i,j)[2] = 255;
                newImg.at<cv::Vec4b>(i,j)[3] = 255;
            } 
            if (localMax.at<uint8_t>(i,j) != 0)
            {
                newImg.at<cv::Vec4b>(i,j)[0] = 255;
                newImg.at<cv::Vec4b>(i,j)[1] = 255;
                newImg.at<cv::Vec4b>(i,j)[2] = 255;
                newImg.at<cv::Vec4b>(i,j)[3] = 255;
            }            
        } 
    }
    cv::imwrite("output/"+file+"_boundaries.tif",newImg);
}

void Segmentation::evaluate(cv::Mat& image, cv::Mat& SLICMask, cv::Mat& localMax, int numCells) //sorensen-dice for watershed
{
    std::set<int> isTree;

    std::set<int> cells;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (cells.find(image.at<int>(i,j)) == cells.end())
            {
                cells.insert(image.at<int>(i,j));
            }
        }      
    }

    std::cout << cells.size() << " " << numCells << std::endl;

    std::vector<cv::Point> coords[5000];
    for (int i = 0; i < localMax.rows; i++)
    {
        for (int j = 0; j < localMax.cols; j++)
        {
            if (image.at<int>(i,j) != -1);
                coords[image.at<int>(i,j)].push_back(cv::Point(i,j));
            if (localMax.at<uint8_t>(i,j) == 255)
            {
                isTree.insert(image.at<int>(i,j));
            }
        }      
    }

    cv::Mat finalMask = cv::Mat::zeros(localMax.size(),CV_8U);
    for (int i = 0; i < finalMask.rows; i++)
    {
        for (int j = 0; j < finalMask.cols; j++)
        {
            if (isTree.find(image.at<int>(i,j)) != isTree.end())
            {
                finalMask.at<uint8_t>(i,j) = 255;
            }          
        }      
    }

    cv::imwrite("output/"+file+"_FinalBound.tif",finalMask);

    cv::Mat groundTruth = cv::imread("data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(finalMask,groundTruth,file);
    e.getSorensenDice(isTree,coords);
}

void Segmentation::evaluate(cv::Mat& mask, std::set<int>& isTree, std::vector<cv::Point>* coord) //soresen-dice for slic
{
    cv::Mat groundTruth = cv::imread("data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(mask,groundTruth,file);
    e.getSorensenDice(isTree,coord);
}

void Segmentation::evaluate(cv::Mat& mask) //overall IOU
{
    cv::Mat groundTruth = cv::imread("data/"+file+"/mask.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(mask,groundTruth,file);
    e.getIOU();
}

void Segmentation::evaluatePer(cv::Mat& mask, int numCells) //IOU per tree
{
    cv::Mat groundTruth_centres = cv::imread("data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    cv::Mat groundTruth_mask = cv::imread("data/"+file+"/mask.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth_centres.data || !groundTruth_mask.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(file);
    e.getIOUPerTree(mask,groundTruth_centres,groundTruth_mask, numCells);
}