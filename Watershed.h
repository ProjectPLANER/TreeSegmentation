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

    static int allocWSNodes( std::vector<WSNode>& storage )
    {
        int sz = (int)storage.size();
        int newsz = MAX(128, sz*3/2);

        storage.resize(newsz);
        if( sz == 0 )
        {
            storage[0].next = 0;
            sz = 1;
        }
        for( int i = sz; i < newsz-1; i++ )
            storage[i].next = i+1;
        storage[newsz-1].next = 0;
        return sz;
    }

    
    private:
        cv::Mat image;
    public:
        Watershed();
        ~Watershed();
        Watershed(cv::Mat& _image);
        void applyWatershed();
        std::vector<cv::Point> bhContoursCenter(const std::vector<std::vector<cv::Point>>& contours,bool centerOfMass);
        std::vector<cv::Point> bhFindLocalMaximum(cv::Mat& src);
        void water(cv::Mat& src, cv::Mat& markers);
        //static int allocWSNodes( std::vector<WSNode>& storage );
};

#endif