/**
 * Lynolan Moodley
 * CSC4002W Project
 */

#ifndef IMAGE_H
#define IMAGE_H

class Image
{
    private:
        cv::Mat dem;
        std::vector<cv::Point2d> maskPoints;
    public:
        Image();
        ~Image();
        Image(std::string dem);
        Image(std::string dem, std::string mask);
        void applyMask(std::string s);
};

#endif