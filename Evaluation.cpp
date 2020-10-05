/**
 * @file Evaluation.cpp
 * @author Lynolan Moodley (mdllyn007@myuct.ac.za)
 * @brief The evaluation definition file, which defines all methods requires to evaluate tree masks after segmentation has taken place.
 * @version 0.1
 * @date 2020-10-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <set>
#include <cmath>
#include <fstream>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Evaluation.h"

/**
 * @brief Construct a new Evaluation:: Evaluation object
 * 
 */
Evaluation::Evaluation() {}

/**
 * @brief Construct a new Evaluation:: Evaluation object
 * 
 * @param file File name of the DEM
 */
Evaluation::Evaluation(std::string file) 
{
    this->file = file;
}

/**
 * @brief Destroy the Evaluation:: Evaluation object
 * 
 */
Evaluation::~Evaluation() {}

/**
 * @brief Calculates the Sorensen-Dice coefficient for a tree mask.
 * 
 * @param groundTruth The ground truth mask
 * @param mask The calculated mask
 * @param isTree A set that contains the ID of clusters that are trees
 * @param coord An array containing a list of pixels that belong to a certain cluster (the index of the array represents a cluster ID)
 */
void Evaluation::getSorensenDice(cv::Mat groundTruth, cv::Mat mask, std::set<int>& isTree, std::vector<cv::Point>* coord) //float for robs, uint8_t for orchard
{
    float matches = 0;
    float falseNegatives = 0;
    float falsePositive = 0;
    for (size_t i = 0; i < groundTruth.rows; i++)
    {
        for (size_t j = 0; j < groundTruth.cols; j++)
        {
            if(groundTruth.at<float>(i,j) == 255 && mask.at<uint8_t>(i,j) == 255)
            {
                matches++;
            }
            else if(groundTruth.at<float>(i,j) == 255 && mask.at<uint8_t>(i,j) != 255)
            {
                falseNegatives++;
            }
        }
    }

    for (std::set<int>::iterator it = isTree.begin(); it != isTree.end(); ++it)
    {
        bool found = false;
        for (int i = 0; i < coord[*it].size(); i++)
        {
            if(groundTruth.at<float>(coord[*it][i].x,coord[*it][i].y) == 255)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            falsePositive++;
        }
    }

    float sorensenDice = (2*matches)/(2*matches + falsePositive + falseNegatives);
    //std::cout << file << std::endl;
    std::cout << "Matches: " << matches << std::endl;
    std::cout << "False positives: " << falsePositive << std::endl;
    std::cout << "False negatives: " << falseNegatives << std::endl;
    std::cout << "Sorensen-Dice coefficient: " << sorensenDice << std::endl;
}

/**
 * @brief Calculates the overall IOU of a tree mask.
 * 
 * @param groundTruth The ground truth mask
 * @param mask The calculated mask
 */
void Evaluation::getIOU(cv::Mat groundTruth, cv::Mat mask)
{
    float I = 0;
    float U = 0;

    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            if (mask.at<uint8_t>(i,j) == 255 && groundTruth.at<float_t>(i,j) == 255)
                I++;
            if (mask.at<uint8_t>(i,j) == 255 || groundTruth.at<float_t>(i,j) == 255)
                U++;            
        }
    }
    
    float IOU = I/U;
    std::cout << "Overall IOU: " << IOU << std::endl;

    /*std::ofstream ofs;
    ofs.open(file+"_localSize.csv",std::ios::app);
    ofs << IOU << std::endl;
    ofs.close();*/
}

/**
 * @brief Calculates the IOU per tree in a tree mask using bounding boxes. Calls getIOUIndividual(std::vector<int>& minmaxTruth, std::vector<int>& minmaxPredicted)
 * 
 * @param groundTruth_mask The ground truth mask
 * @param groundTruth_centres The ground tree centres
 * @param mask The calculated mask
 * @param numClusters The number of clusters found in the image
 */
void Evaluation::getIOUPerTree(cv::Mat& groundTruth_mask, cv::Mat& groundTruth_centres, cv::Mat& mask, int numClusters)
{
    std::vector<cv::Point> clusters[5000];
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {         
            if (mask.at<int>(i,j) != -1)
            {
                clusters[mask.at<int>(i,j)].push_back(cv::Point(i,j));
            }
                
        }
    }

    std::vector<int> minMaxPredicted[numClusters];
    for (int i = 0; i < numClusters; i++)
    {
        int minX = INT32_MAX;
        int maxX = INT32_MIN;
        int minY = INT32_MAX;
        int maxY = INT32_MIN;
        for (int j = 0; j < clusters[i].size(); j++)
        {
            if (clusters[i][j].x < minX)
                minX = clusters[i][j].x;
            if (clusters[i][j].x > maxX)
                maxX = clusters[i][j].x;
            if (clusters[i][j].y < minY)
                minY = clusters[i][j].y;
            if (clusters[i][j].y > maxY)
                maxY = clusters[i][j].y;
        }
        minMaxPredicted[i].push_back(minX);
        minMaxPredicted[i].push_back(maxX);
        minMaxPredicted[i].push_back(minY);
        minMaxPredicted[i].push_back(maxY);
    }

    std::vector<std::vector<int>> minMaxTruth;
    for (int i = 0; i < groundTruth_centres.rows; i++)
    {
        for (int j = 0; j < groundTruth_centres.cols; j++)
        {
            if (groundTruth_centres.at<float_t>(i,j) == 255) //14 for 30px wide; 7 for 15px
            {
                int minX = i - 14;
                int maxX = i + 14;
                int minY = j - 14;
                int maxY = j + 14;
                
                std::vector<int> c;
                c.push_back(minX);
                c.push_back(maxX);
                c.push_back(minY);
                c.push_back(maxY);
                minMaxTruth.push_back(c);
            }
        }
    }
    
    std::vector<float> IOU;
    std::vector<int> bestCluster;
    for (int i = 0; i < minMaxTruth.size(); i++)
    {
        float maxIOU = 0;
        int best = 0;
        for (int j = 0; j < numClusters; j++)
        {
            float currIOU = getIOUIndividual(minMaxTruth[i],minMaxPredicted[j]);
            if (maxIOU < currIOU)
            {
                maxIOU = currIOU;
                best = j;
            }
        }
        IOU.push_back(maxIOU);
        bestCluster.push_back(best);
    }
    
    int bucket[10];
    for (size_t i = 0; i < 10; i++)
    {
        bucket[i] = 0;
    }

    float mean = 0;

    for (int i = 0; i < IOU.size(); i++)
    {
        mean += IOU[i];
        if (IOU[i] < 0.1)
        {
            bucket[0]++;
        }
        else if (IOU[i] < 0.2)
        {
            bucket[1]++;
        }
        else if (IOU[i] < 0.3)
        {
            bucket[2]++;
        }
        else if (IOU[i] < 0.4)
        {
            bucket[3]++;
        }
        else if (IOU[i] < 0.5)
        {
            bucket[4]++;
        }
        else if (IOU[i] < 0.6)
        {
            bucket[5]++;
        }
        else if (IOU[i] < 0.7)
        {
            bucket[6]++;
        }
        else if (IOU[i] < 0.8)
        {
            bucket[7]++;
        }
        else if (IOU[i] < 0.9)
        {
            bucket[8]++;
        }
        else
        {
            bucket[9]++;
        }       
    } 

    /*std::ofstream ofs("output/"+file+"_buckWater.csv");
    for (size_t i = 0; i < 10; i++)
    {
        ofs << buck[i] << ",";
    }
    ofs.close();*/

    mean /= IOU.size();
    float stddev = 0;
    for (int i = 0; i < IOU.size(); i++)
    {
        stddev += (std::pow((IOU[i]-mean),2));
    }

    stddev = std::sqrt(stddev/(IOU.size()-1));
    std::cout << "Mean IOU: " << mean << std::endl;
    std::cout << "STD DEV: " << stddev << std::endl;

    /*std::ofstream ofs2;
    ofs2.open("timestdc.csv",std::ios::app);
    ofs2 << mean << "," << stddev << std::endl;
    ofs2.close();*/
}

/**
 * @brief Calculates the IOU of a single tree.
 * 
 * @param minmaxTruth The minimum and maximum coordinate values of the ground truth tree
 * @param minmaxPredicted The minimum and maximum coordinate values of the calculated tree
 * @return float 
 */
float Evaluation::getIOUIndividual(std::vector<int>& minmaxTruth, std::vector<int>& minmaxPredicted)
{
    if (minmaxTruth[0] > minmaxPredicted[1] || minmaxPredicted[0] > minmaxTruth[1] || minmaxTruth[2] > minmaxPredicted[3] || minmaxPredicted[2] > minmaxTruth[3])
    {
        return 0;
    }

    float areaTruth = (minmaxTruth[1] - minmaxTruth[0] + 1) * (minmaxTruth[3] - minmaxTruth[2] + 1) - 85; //85 for 30px wide; 40 for 15px wide
    float areaPredicted = (minmaxPredicted[1] - minmaxPredicted[0] + 1) * (minmaxPredicted[3] - minmaxPredicted[2] + 1);

    float areaI = (std::min(minmaxTruth[1], minmaxPredicted[1]) - std::max(minmaxTruth[0], minmaxPredicted[0]) + 1) * (std::min(minmaxTruth[3], minmaxPredicted[3]) - std::max(minmaxTruth[2], minmaxPredicted[2]) + 1);

    float IOU = areaI / (areaPredicted + areaTruth - areaI);
    return IOU;
}

/**
 * @brief Calculates the Euclidean distance between every ground truth centre and the nearest local maximum.
 * 
 * @param groundTruth_centres The ground truth of tree centres
 * @param localMax The calculated local maxima
 */
void Evaluation::getCentreOffsets(cv::Mat& groundTruth_centres, cv::Mat& localMax)
{
    std::vector<float> dist;
    for (int i = 0; i < groundTruth_centres.rows; i++)
    {
        for (int j = 0; j < groundTruth_centres.cols; j++)
        {     
            if (groundTruth_centres.at<float>(i,j) == 255)
            {
                float minDist = groundTruth_centres.rows;
                for (int k = 0; k < localMax.rows; k++)
                {
                    for (int l = 0; l < localMax.cols; l++)
                    {
                        if (localMax.at<uint8_t>(k,l) == 255)
                        {
                            minDist = std::min(minDist,getEuclideanDist(i,k,j,l));
                        }
                    }
                }
                dist.push_back(minDist);
            }
        }
    }

    float mean = 0;
    int count = 0;
    std::vector<float> con;
    for (size_t i = 0; i < dist.size(); i++)
    {
        if (dist[i] <= 15)
        {
            mean += dist[i];
            count++;
            con.push_back(dist[i]);
        }
    }
    mean /= count;
    float stddev = 0;
    for (int i = 0; i < count; i++)
    {
        stddev += (std::pow((con[i]-mean),2));
    }
    stddev = std::sqrt(stddev/(con.size()-1));

    std::cout << "Mean dist: " << mean << std::endl;
    std::cout << "STD DEV: " << stddev << std::endl;

    /*std::ofstream ofs;
    ofs.open("localMaxDist3.csv",std::ios::app);
    ofs << dist.size() << "," << mean << "," << stddev << "," << count << "," << dist.size()-count << std::endl;
    ofs.close();*/
}

/**
 * @brief Calculates the Euclidean distance between 2 pixels.
 * 
 * @param x1 The x coordinate of pixel 1
 * @param x2 The x coordinate of pixel 2
 * @param y1 The y coordinate of pixel 1
 * @param y2 The y coordinate of pixel 2
 * @return float 
 */
float Evaluation::getEuclideanDist(int x1, int x2, int y1, int y2)
{
    return std::sqrt(std::pow((x2-x1),2)+std::pow((y2-y1),2));
}