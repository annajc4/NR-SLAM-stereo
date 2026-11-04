#include "datasets/stereomis.h"
#include "SLAM/system.h"

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/log/check.h"

using namespace std;

ABSL_FLAG(std::string, sequence_path, "", "StereoMIS sequence folder (with video_frames/)");
ABSL_FLAG(std::string, settings_path, "", "Path to the settings file");
ABSL_FLAG(int, starting_frame, 1, "First file number to process");
ABSL_FLAG(int, end_frame, 0, "Stop before this file number");
ABSL_FLAG(int, step, 2, "Use every step-th file (StereoMIS sequences.txt)");

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    string sequence_path = absl::GetFlag(FLAGS_sequence_path);
    if(sequence_path.empty()){
        LOG(ERROR) << "Must specify an input dataset path." << endl;
        return -1;
    }

    string settings_path = absl::GetFlag(FLAGS_settings_path);
    if(settings_path.empty()){
        LOG(ERROR) << "Must specify an input settings file." << endl;
        return -1;
    }

    int starting_frame = absl::GetFlag(FLAGS_starting_frame);
    int end_frame = absl::GetFlag(FLAGS_end_frame);
    int step = absl::GetFlag(FLAGS_step);

    StereoMIS dataset(sequence_path);
    System SLAM(settings_path);

    for (int idx = starting_frame; idx < end_frame; idx += step) {
        LOG(INFO) << "Processing image " << idx;
        auto im_left = dataset.GetImage(idx);
        CHECK_OK(im_left);

        auto im_right = dataset.GetRightImage(idx);
        CHECK_OK(im_right);

        SLAM.TrackImageWithStereo(*im_left, *im_right);
    }

    return 0;
}