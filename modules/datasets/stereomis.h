#ifndef NRSLAM_STEREOMIS_H
#define NRSLAM_STEREOMIS_H

#include <string>

#include "absl/status/statusor.h"

#include <opencv2/opencv.hpp>

class StereoMIS {
public:
    //uses the video frames folder that the converter stereomis.py already ran 
    StereoMIS(const std::string& sequence_path);

    absl::StatusOr<cv::Mat> GetImage(const int idx);

    absl::StatusOr<cv::Mat> GetRightImage(const int idx);
    
    absl::StatusOr<cv::Mat> GetMask(const int idx);

    absl::StatusOr<double> GetTimestamp(const int idx);

    absl::StatusOr<cv::Mat> GetLiverMask(const int idx);
private:
    absl::StatusOr<cv::Mat> ReadImage(const int idx, const char side);

    std::string frames_path_;

    std::string masks_path_;

    std::string liver_masks_path_;

    std::vector<double> times_;

};

#endif //NRSLAM_STEREOMIS_H
