#include "stereomis.h"
#include <fstream>

using namespace std;

StereoMIS::StereoMIS(const std::string& sequence_path) : frames_path_(sequence_path + "/video_frames/"),
                                                         masks_path_(sequence_path + "/masks/") {
    ifstream file(frames_path_ + "times.txt");
    double time;
    while (file >> time) {
        times_.push_back(time);
    }
}

absl::StatusOr<cv::Mat> StereoMIS::GetImage(const int idx) {
    return ReadImage(idx, 'l');
}

absl::StatusOr<cv::Mat> StereoMIS::GetRightImage(const int idx) {
    return ReadImage(idx, 'r');
}

absl::StatusOr<cv::Mat> StereoMIS::ReadImage(const int idx, const char side) {
    string name = cv::format("%06d%c.png", idx, side);
    cv::Mat image = cv::imread(frames_path_ + name, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return absl::NotFoundError("No image " + frames_path_ + name);
    }
    return image;
}

absl::StatusOr<cv::Mat> StereoMIS::GetMask(const int idx) {
    string name = cv::format("%06dl.png", idx);
    cv::Mat mask = cv::imread(masks_path_ + name, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        return absl::NotFoundError("No mask " + masks_path_ + name);
    }
    return mask;
}

absl::StatusOr<double> StereoMIS::GetTimestamp(const int idx) {
    if (idx < 1 || idx > times_.size()) {
        return absl::NotFoundError("No timestamp for file " + to_string(idx));
    }
    return times_[idx - 1];
}


