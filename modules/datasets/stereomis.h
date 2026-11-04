#ifndef NRSLAM_STEREOMIS_H
#define NRSLAM_STEREOMIS_H

#include <string>

#include "absl/status/statusor.h"

#include <opencv2/opencv.hpp>

class StereoMIS {
public:
    //uses the video frames folder that the converter stereomis.py already ran 
    StereoMIS(const std::string& sequence_path);

    absl::StatusOr<cv::Mat> GetImage(const int n);

    absl::StatusOr<cv::Mat> GetRightImage(const int n);

private:
    absl::StatusOr<cv::Mat> ReadImage(const int n, const char side);

    std::string frames_path_;
};

#endif //NRSLAM_STEREOMIS_H
