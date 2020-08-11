/**
 * Lynolan Moodley
 * CSC4002W_Project
 */

#ifndef WATERSHED
#define WATERSHED

class Watershed
{
    struct WSNode
    {
        int next;
        int mask_ofs;
        int img_ofs;
    };

    // Queue for WSNodes
    struct WSQueue
    {
        WSQueue() { first = last = 0; }
        int first, last;
    };
    private:
        cv::Mat image;
    public:
        Watershed();
        ~Watershed();
        Watershed(cv::Mat& _image);
        void applyWatershed();
        std::vector<cv::Point> bhContoursCenter(const std::vector<std::vector<cv::Point>>& contours,bool centerOfMass);
        std::vector<cv::Point> bhFindLocalMaximum(cv::Mat& src);
        void water(cv::InputArray _src, cv::InputOutputArray _markers);
};

#endif