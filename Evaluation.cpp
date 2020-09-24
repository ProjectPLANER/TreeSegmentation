/**
 * Lynolan Moodley
 * CSC4002W Project
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <set>
#include "Evaluation.h"

#include <fstream>
#include <iostream>

Evaluation::Evaluation() {}

Evaluation::Evaluation(std::string file) 
{
    this->file = file;
}

Evaluation::Evaluation(cv::Mat& image, cv::Mat& groundTruth, std::string file)
{
    this->image = image;
    this->groundTruth = groundTruth;
    this->file = file;
}

Evaluation::~Evaluation() {}

/**
 * @brief 
 * 
 */
void Evaluation::getSorensenDice(std::set<int>& isTree, std::vector<cv::Point>* coord) //float for robs, uint8_t for orchard
{
    float matches = 0;
    float falseNegatives = 0;
    float falsePositive = 0;
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {
            if(groundTruth.at<float>(i,j) == 255 && image.at<uint8_t>(i,j) == 255)
            {
                matches++;
            }
            else if(groundTruth.at<float>(i,j) == 255 && image.at<uint8_t>(i,j) != 255)
            {
                falseNegatives++;
            }
        }
    }

    for (std::set<int>::iterator it = isTree.begin(); it != isTree.end(); ++it)
    {
        bool found = false;
        for (size_t i = 0; i < coord[*it].size(); i++)
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
    std::cout << file << std::endl;
    std::cout << "Matches: " << matches << std::endl;
    std::cout << "False positives: " << falsePositive << std::endl;
    std::cout << "False negatives: " << falseNegatives << std::endl;
    std::cout << "Sorensen-Dice coefficient: " << sorensenDice << std::endl;
}

void Evaluation::getIOU()
{
    float I = 0;
    float U = 0;

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (image.at<uint8_t>(i,j) == 255 && groundTruth.at<float_t>(i,j) == 255)
                I++;
            if (image.at<uint8_t>(i,j) == 255 || groundTruth.at<float_t>(i,j) == 255)
                U++;            
        }
    }
    
    float IOU = I/U;
    std::cout << "IOU: " << file << " " << IOU << std::endl;
}

void Evaluation::getIOUPerTree(cv::Mat& mask, cv::Mat& groundTruth_centres, cv::Mat& groundTruth_mask, int numCells)
{
    cv::imwrite("betty.tif",mask);
    //mask = mask + 1;
    std::vector<cv::Point> cells[5000];
    std::cout << numCells << std::endl;
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {         
            if (mask.at<int>(i,j) != -1)
            {
                std::cout << mask.at<int>(i,j) << std::endl;
                cells[mask.at<int>(i,j)].push_back(cv::Point(i,j));
                //std::cout << j << std::endl;
            }
                
        }
    }

    std::vector<int> minMaxPredicted[numCells];
    for (int i = 0; i < numCells; i++)
    {
        int minX = INT32_MAX;
        int maxX = INT32_MIN;
        int minY = INT32_MAX;
        int maxY = INT32_MIN;
        for (int j = 0; j < cells[i].size(); j++)
        {
            if (cells[i][j].x < minX)
                minX = cells[i][j].x;
            if (cells[i][j].x > maxX)
                maxX = cells[i][j].x;
            if (cells[i][j].y < minY)
                minY = cells[i][j].y;
            if (cells[i][j].y > maxY)
                maxY = cells[i][j].y;
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
    std::vector<int> bestCell;
    for (int i = 0; i < minMaxTruth.size(); i++)
    {
        float maxIOU = 0;
        int best = 0;
        for (int j = 0; j < numCells; j++)
        {
            //maxIOU = std::max(maxIOU,getIOUIndividual(minMaxTruth[i],minMaxPredicted[j]));
            float currIOU = getIOUIndividual(minMaxTruth[i],minMaxPredicted[j]);
            if (maxIOU < currIOU)
            {
                maxIOU = currIOU;
                best = j;
            }
        }
        IOU.push_back(maxIOU);
        bestCell.push_back(best);
    }
    
    int buck[10];
    for (size_t i = 0; i < 10; i++)
    {
        buck[i] = 0;
    }
    
    std::cout << "IOU" << std::endl;
    std::cout << "===" << std::endl;
    for (int i = 0; i < IOU.size(); i++)
    {
        std::cout << i << ": " << bestCell[i] << " " << IOU[i] << std::endl;
        if (IOU[i] < 0.1)
        {
            buck[0]++;
        }
        else if (IOU[i] < 0.2)
        {
            buck[1]++;
        }
        else if (IOU[i] < 0.3)
        {
            buck[2]++;
        }
        else if (IOU[i] < 0.4)
        {
            buck[3]++;
        }
        else if (IOU[i] < 0.5)
        {
            buck[4]++;
        }
        else if (IOU[i] < 0.6)
        {
            buck[5]++;
        }
        else if (IOU[i] < 0.7)
        {
            buck[6]++;
        }
        else if (IOU[i] < 0.8)
        {
            buck[7]++;
        }
        else if (IOU[i] < 0.9)
        {
            buck[8]++;
        }
        else
        {
            buck[9]++;
        }       
    } 

    std::ofstream ofs("output/"+file+"_buck.csv");
    for (size_t i = 0; i < 10; i++)
    {
        ofs << buck[i] << ",";
    }
    ofs.close();

}

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