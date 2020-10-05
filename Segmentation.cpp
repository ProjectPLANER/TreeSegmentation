/**
 * @file Segmentation.cpp
 * @author Lynolan Moodley (mdllyn007@myuct.ac.za)
 * @brief The segmentation definition file, which defines the methods to be used when segmenting DEMs.
 * @version 0.1
 * @date 2020-10-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <set>
#include <chrono>
#include <fstream>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Preprocessing.h"
#include "Evaluation.h"
#include "Segmentation.h"

/**
 * @brief Construct a new Segmentation:: Segmentation object
 * 
 * @param image The DEM
 * @param file The file name (without directory or file extension)
 * @param min The minimum pixel value in the DEM
 * @param SLICSize The SLIC cluster size
 * @param localMaxWindowSize The local maxima window size
 */
Segmentation::Segmentation(cv::Mat& image, std::string file, int min, int SLICSize, int localMaxWindowSize) 
{
    this->image = image;
    this->file = file;
    this->min = min;
    this->SLICSize = SLICSize;
    this->localMaxWindowSize = localMaxWindowSize;
}

/**
 * @brief Destroy the Segmentation:: Segmentation object
 * 
 */
Segmentation::~Segmentation() {}

/**
 * @brief Performs the segmentation operation on a DEM.
 * 
 */
void Segmentation::segment(bool test)
{   
    //auto start = std::chrono::high_resolution_clock::now();
    Preprocessing p;

    //Create a mask image of the DEM
    cv::Mat mask = p.applyThreshold(image,min,1); //easy: (image,0,1); medium:(image,-2,1); hard:(image,19,1)
    mask.convertTo(mask,CV_8U);
    std::cout << "Mask image created..." << std::endl;

    //If the DEM is not rectangular, set the null value to 0
    p.removeBoundary(image,min); //16 for easy
    std::cout << "Boundary removed..." << std::endl;

    //Calculate the local maxima in the DEM
    cv::Mat localMax = p.findLocalMax(image, localMaxWindowSize);
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
    
    //cv::imwrite(file+"_8UC1.tif",image_8U);
    image_8U = p.applyFilterGausian(image_8U);
    //cv::imwrite(file+"_8UC1BIL.tif",image_8U);

    cv::Mat slicmask, result;

    //Time
    auto startSLIC = std::chrono::high_resolution_clock::now();

    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(image,cv::ximgproc::SLICO,SLICSize,0.95); //30,10,   28
    slic->iterate(6);
    slic->getLabelContourMask(slicmask, true);
    cv::Mat clusters = cv::Mat::zeros(image_8U.size(),CV_32S);
    std::cout << slic->getNumberOfSuperpixels() << " superpixel clusters found..." << std::endl;
    slic->getLabels(clusters);

    auto stopSLIC = std::chrono::high_resolution_clock::now();
    auto durationSLIC = std::chrono::duration_cast<std::chrono::milliseconds>(stopSLIC - startSLIC);
    //std::cout << "Time: " << durationSLIC.count() << std::endl;

    /*std::ofstream ofsLocalSize;
    ofsLocalSize.open(file+"_localSize.csv",std::ios::app);
    ofsLocalSize << ite << ",";
    ofsLocalSize.close();*/

    /* SLIC test
    std::vector<cv::Vec3b> SLICColours;
    for (size_t i = 0; i < slic->getNumberOfSuperpixels(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        SLICColours.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat SLICdst = cv::Mat::zeros(clusters.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < clusters.rows; i++)
    {
        for (int j = 0; j < clusters.cols; j++)
        {
            int index = clusters.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(slic->getNumberOfSuperpixels()))
            {
                SLICdst.at<cv::Vec3b>(i,j) = SLICColours[index-1];
            }
        }
    }
    cv::imwrite("output/"+file+"_SLIC.tif",SLICdst);*/

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
            avgVar += p.getVariance(mapping[i]);
            numTrees++;        
        }
    }
    avgVar = avgVar/numTrees;
    float thresVar = avgVar * 0.1;

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
            
            if ((count == 1 || count == 2 || count == 3) && p.getVariance(mapping[i]) > thresVar)
            {
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
    //cv::imwrite("treeSlicMask.tif",treeSLICMask);
    //cv::Mat edges = p.findEdges(image_8U);
    //cv::imwrite("edges.tif",edges);

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
    
    //cv::imwrite(file+"_finalImg.tif",newImg);

/*  Region growing test
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

    //cv::imwrite("markers.tif",markers);
    //cv::imwrite("newImg.tif",newImg);

    //TIME
    //auto startWater = std::chrono::high_resolution_clock::now();
    cv::watershed(newImg,markers); //image_8UC3
    //auto stopWater = std::chrono::high_resolution_clock::now();
    //auto durationWater = std::chrono::duration_cast<std::chrono::milliseconds>(stopWater - startWater); 
    std::cout << "Watershed complete..." << std::endl;
    //cv::imwrite("output/"+file+"_watershedLines.tif",markers);

    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    
    //Generate random colours for the output mask
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
    cv::imwrite("../output/"+file+"/watershedBasins.tif",dst);
    cv::Mat watershedMask = getWatershedMask(mark);

    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    //recordTime(duration.count(),durationSLIC.count(),durationWater.count());

    // Visualize the final image
    display(image_8UC3,dst,slicmask,localMax);

    if (test)
    {
        evaluateSorensenDice(markers, localMax, contours.size());
        evaluateIOU(watershedMask);
        evaluateIOUPerTree(markers, contours.size());
        evaluateCentre(localMax);
    }

    //evaluate(treeSLICMask,isTree,coord);
    //evaluate(watershedMask);
    //evaluate(markers,treeSLICMask,localMax,contours.size());
    //evaluateIOUPerTree(markers,contours.size());
    //evaluateIOUPerTreeSLIC(clusters,isTree,slic->getNumberOfSuperpixels());
    //evaluateCentre(localMax);

    cv::namedWindow("Final Result",cv::WINDOW_NORMAL);
    cv::resizeWindow("Final Result", 1920,1080);
    cv::imshow("Final Result", dst);
    cv::waitKey(0);
}

/**
 * @brief Produces a binary mask for the segmented DEM.
 * 
 * @param image An image containing clusters
 * @return cv::Mat 
 */
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
    cv::imwrite("../output/"+file+"/watershedMask.tif",watershedMask);
    return watershedMask;
}

/**
 * @brief Creates an image showing the effect of the different segmentation techniques.
 * 
 * @param image The original DEM
 * @param boundaries Watershed clusters
 * @param slicBoundaries SLIC clusters
 * @param localMax Local maxima calculated
 */
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
    cv::imwrite("../output/"+file+"/boundaries.tif",newImg);
}

/**
 * @brief Records the execution times calculated for different operations.
 * 
 * @param duration Total time
 * @param durationSLIC SLIC execution time
 * @param durationWater Watershed execution time
 */
void Segmentation::recordTime(int duration, int durationSLIC, int durationWater)
{
    std::ofstream ofs("output/"+file+"_time.csv");
    ofs << duration << "," << durationSLIC << "," << durationWater;
    ofs.close();
}

/**
 * @brief Calculates the Sorensen-Dice coefficient for the watershed segmentation.
 * 
 * @param image An image containing the watershed clusters
 * @param localMax The local maxima calculated
 * @param numClusters The number of watershed clusters calculated
 */
void Segmentation::evaluateSorensenDice(cv::Mat& image, cv::Mat& localMax, int numClusters) //sorensen-dice for watershed
{
    std::set<int> isTree;

    std::set<int> clusters;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (clusters.find(image.at<int>(i,j)) == clusters.end())
            {
                clusters.insert(image.at<int>(i,j));
            }
        }      
    }

    std::cout << clusters.size() << " " << numClusters << std::endl;

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

    //cv::imwrite("output/"+file+"_FinalBound.tif",finalMask);

    cv::Mat groundTruth = cv::imread("../data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(file);
    e.getSorensenDice(groundTruth, finalMask, isTree, coords);
}

/**
 * @brief Calculates the Sorensen-Dice coefficient for the SLIC segmentation.
 * 
 * @param mask An image containing the SLIC clusters
 * @param isTree A set containg the IDs of clusters that are trees
 * @param coord An array containing a list of all pixels that below to each cluster
 */
void Segmentation::evaluateSorensenDiceSLIC(cv::Mat& mask, std::set<int>& isTree, std::vector<cv::Point>* coord) //soresen-dice for slic
{
    cv::Mat groundTruth = cv::imread("../data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(file);
    e.getSorensenDice(groundTruth, mask, isTree, coord);
}

/**
 * @brief Calculates the overall IOU.
 * 
 * @param mask An input mask of trees
 */
void Segmentation::evaluateIOU(cv::Mat& mask) //overall IOU
{
    cv::Mat groundTruth = cv::imread("../data/"+file+"/mask.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(file);
    e.getIOU(groundTruth, mask);
}

/**
 * @brief Calculates the IOU per tree.
 * 
 * @param clusters An image containing segmentation clusters
 * @param numClusters The number of clusters in the image 
 */
void Segmentation::evaluateIOUPerTree(cv::Mat& clusters, int numClusters) //IOU per tree
{
    cv::Mat groundTruth_centres = cv::imread("../data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    cv::Mat groundTruth_mask = cv::imread("../data/"+file+"/mask.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth_centres.data || !groundTruth_mask.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    Evaluation e(file);
    e.getIOUPerTree(groundTruth_mask, groundTruth_centres, clusters, numClusters);
}

/**
 * @brief Calculates the IOU per tree for SLIC segmentation.
 * 
 * @param clusters An image containing segmentation clusters
 * @param isTree A set containg the IDs of clusters that are trees
 * @param numClusters The number of clusters in the image
 */
void Segmentation::evaluateIOUPerTreeSLIC(cv::Mat& clusters, std::set<int>& isTree, int numClusters) //IOU per tree slic
{
    cv::Mat newCluster = clusters.clone();
    cv::Mat groundTruth_centres = cv::imread("../data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    cv::Mat groundTruth_mask = cv::imread("../data/"+file+"/mask.tif",cv::IMREAD_UNCHANGED);
    if(!groundTruth_centres.data || !groundTruth_mask.data)
	{
		std::cout << "Error: Could not open or locate the ground truth file." << std::endl;
        return;
	}

    for (int i = 0; i < clusters.rows; i++)
    {
        for (int j = 0; j < clusters.cols; j++)
        {
            if (isTree.find(clusters.at<int>(i,j)) == isTree.end())
            {
                newCluster.at<int>(i,j) = -1;
            }
        }   
    }
    Evaluation e(file);
    e.getIOUPerTree(newCluster,groundTruth_centres,groundTruth_mask, numClusters);
}

/**
 * @brief Calculates the Euclidean distance between the ground truth centres and the nearest local maximum calculated.
 * 
 * @param localMax The local maxima calculated
 */
void Segmentation::evaluateCentre(cv::Mat& localMax) //Centre offset
{
    cv::Mat groundTruth_centres = cv::imread("../data/"+file+"/centres.tif",cv::IMREAD_UNCHANGED);
    Evaluation e(file);
    e.getCentreOffsets(groundTruth_centres, localMax);
}