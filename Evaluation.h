/**
 * @file Evaluation.h
 * @brief The evaluation header, which declares all methods requires to evaluate tree masks after segmentation has taken place.
 * @version 0.1
 * @date 2020-10-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef EVALUATION_H
#define EVALUATION_H

class Evaluation
{
    private:
        std::string file;
        float getIOUIndividual(std::vector<int>& minmaxTruth, std::vector<int>& minmaxPredicted);
        float getEuclideanDist(int x1, int x2, int y1, int y2);
    public:
        Evaluation(cv::Mat& image, cv::Mat& groundTruth, std::string file);
        Evaluation();
        Evaluation(std::string file);
        ~Evaluation();
        void getSorensenDice(cv::Mat groundTruth, cv::Mat mask, std::set<int>& isTree, std::vector<cv::Point>* coord);
        void getIOU(cv::Mat groundTruth, cv::Mat mask);
        void getIOUPerTree(cv::Mat& groundTruth_mask, cv::Mat& groundTruth_centres, cv::Mat& mask, int numClusters);
        void getCentreOffsets(cv::Mat& groundTruth_centres, cv::Mat& localMax);
};

#endif
