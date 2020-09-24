/**
 * Lynolan Moodley
 * CSC4002W Project
 */

#ifndef EVALUATION_H
#define EVALUATION_H

class Evaluation
{
    private:
        cv::Mat image;
        cv::Mat groundTruth;
        std::string file;
        float getIOUIndividual(std::vector<int>& minmaxTruth, std::vector<int>& minmaxPredicted);
    public:
        Evaluation(cv::Mat& image, cv::Mat& groundTruth, std::string file);
        Evaluation();
        Evaluation(std::string file);
        ~Evaluation();
        void getSorensenDice(std::set<int>& isTree, std::vector<cv::Point>* coord);
        void getIOU();
        void getIOUPerTree(cv::Mat& mask, cv::Mat& groundTruth_centres, cv::Mat& groundTruth_mask, int numCells);
};

#endif