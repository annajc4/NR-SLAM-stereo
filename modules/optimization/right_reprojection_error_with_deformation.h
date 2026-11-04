#ifndef NRSLAM_RIGHT_REPROJECTION_ERROR_WITH_DEFORMATION_H
#define NRSLAM_RIGHT_REPROJECTION_ERROR_WITH_DEFORMATION_H

#include "landmark_vertex.h"

#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/vertex_se3_expmap.h"

class RightReprojectionErrorWithDeformation : public g2o::BaseBinaryEdge<1, double, g2o::VertexSE3Expmap, LandmarkVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void computeError() {
        const g2o::VertexSE3Expmap* pose_vertex = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const LandmarkVertex* deformation_vertex = static_cast<const LandmarkVertex*>(_vertices[1]);

        g2o::SE3Quat Tcw = pose_vertex->estimate();
        Eigen::Vector3d deformation = deformation_vertex->estimate();

        Eigen::Vector3d landmark_camera = Tcw.map(deformation + landmark_world_);
        double predicted_right_x = fx_ * landmark_camera.x() / landmark_camera.z()
                           + cx_ - bf_ / landmark_camera.z();

_error(0) = _measurement - predicted_right_x;
    }

    bool read(std::istream& is) { return false; }

    bool write(std::ostream& os) const { return false; }

    Eigen::Vector3d landmark_world_ = Eigen::Vector3d::Zero();   // zero in the BA (vertex = full position)
    double fx_ = 0, cx_ = 0, bf_ = 0;
};

#endif //NRSLAM_RIGHT_REPROJECTION_ERROR_WITH_DEFORMATION_H
