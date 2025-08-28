/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// 在 .cpp 里提供唯一性定义：int voxel_plane_id = 0;
extern int voxel_plane_id;

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

// ---------------------- Forward decls from project ----------------------
struct pointWithVar;

// ---------------------- Config / basic types ----------------------------
struct VoxelMapConfig {
  double max_voxel_size_{};
  int    max_layer_{};
  int    max_iterations_{};
  std::vector<int> layer_init_num_;
  int    max_points_num_{};
  double planner_threshold_{};
  double beam_err_{};
  double dept_err_{};
  double sigma_num_{};
  bool   is_pub_plane_map_{};

  // local map sliding
  double sliding_thresh{};
  bool   map_sliding_en{};
  int    half_map_size{};
};

struct PointToPlane {
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_{};
  double d_{};
  double eigen_value_{};
  bool is_valid_{};
  float dis_to_plane_{};
};

struct VoxelPlane {
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0.f;
  float min_eigen_value_ = 1.f;
  float mid_eigen_value_ = 1.f;
  float max_eigen_value_ = 1.f;
  float d_ = 0.f;
  int   points_size_ = 0;
  bool  is_plane_ = false;
  bool  is_init_  = false;
  int   id_ = 0;
  bool  is_update_ = false;

  VoxelPlane() {
    plane_var_.setZero();
    covariance_.setZero();
    center_.setZero();
    normal_.setZero();
    y_normal_.setZero();
    x_normal_.setZero();
  }
};

class VOXEL_LOCATION {
public:
  int64_t x{0}, y{0}, z{0};
  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}
  bool operator==(const VOXEL_LOCATION& o) const { return x == o.x && y == o.y && z == o.z; }
};

// Hash for VOXEL_LOCATION
namespace std {
template<>
struct hash<VOXEL_LOCATION> {
  size_t operator()(const VOXEL_LOCATION& s) const {
    return static_cast<size_t>((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT {
  float xyz[3]{};
  float intensity{};
  int   count{0};
};

void calcBodyCov(Eigen::Vector3d& pb, const float range_inc, const float degree_inc, Eigen::Matrix3d& cov);

// ---------------------- Octo Tree ---------------------------------------
class VoxelOctoTree {
public:
  VoxelOctoTree() = default;

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
  : plane_ptr_(new VoxelPlane),
    layer_(layer),
    octo_state_(0),
    quater_length_(0.f),
    planer_threshold_(planer_threshold),
    points_size_threshold_(points_size_threshold),
    update_size_threshold_(5),
    max_points_num_(max_points_num),
    max_layer_(max_layer),
    new_points_(0),
    init_octo_(false),
    update_enable_(true) {
    for (int i = 0; i < 8; ++i) leaves_[i] = nullptr;
  }

  ~VoxelOctoTree() {
    for (int i = 0; i < 8; ++i) delete leaves_[i];
    delete plane_ptr_;
  }

  // minimal fields used by functions below
  std::vector<pointWithVar> temp_points_;
  VoxelPlane* plane_ptr_{nullptr};
  int layer_{0};
  int octo_state_{0};                 // 0 leaf, 1 internal
  VoxelOctoTree* leaves_[8]{};
  double voxel_center_[3]{0.0,0.0,0.0};
  std::vector<int> layer_init_num_;
  float  quater_length_{0.f};
  float  planer_threshold_{0.f};
  int    points_size_threshold_{0};
  int    update_size_threshold_{5};
  int    max_points_num_{0};
  int    max_layer_{0};
  int    new_points_{0};
  bool   init_octo_{false};
  bool   update_enable_{true};

  // 声明（实现放在 .cpp，避免与 .cpp 的定义重复）
  void correctPose(const Eigen::Matrix3d& R_corr, const Eigen::Vector3d& t_corr);

  // 仅在头文件内联提供（.cpp 没有再次实现）
  void collectPoints(std::vector<pointWithVar>& out_points) {
    if (octo_state_ == 0) {
      out_points.insert(out_points.end(), temp_points_.begin(), temp_points_.end());
    } else {
      for (int i = 0; i < 8; ++i) if (leaves_[i]) leaves_[i]->collectPoints(out_points);
    }
  }

  // Declarations (implemented elsewhere in project)
  void init_plane(const std::vector<pointWithVar>& points, VoxelPlane* plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar& pv);
  VoxelOctoTree* find_correspond(Eigen::Vector3d pw);
  VoxelOctoTree* Insert(const pointWithVar& pv);
};

// ---------------------- Manager -----------------------------------------
void loadVoxelConfig(ros::NodeHandle& nh, VoxelMapConfig& voxel_config);

class VoxelMapManager {
public:
  VoxelMapManager() = default;

  VoxelMapManager(VoxelMapConfig& config_setting,
                  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& voxel_map)
  : config_setting_(config_setting), voxel_map_(voxel_map) {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  }

  // ---- Minimal getters used by other modules ----
  inline std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*>& getVoxelMap() { return voxel_map_; }

  // 一些地方会拿到一个“octree()”句柄；如果你没有单根节点，这里返回 nullptr 即可（外部已有判空）
  inline VoxelOctoTree* octree() const { return nullptr; }

  // 声明（实现放 .cpp；之前与 .cpp 重复定义导致报错）
  std::vector<VoxelPlane*> getVoxelsInRange(double x, double y, double z, double radius);

  // ---- PGO 会调用这个版本；提供一个小的 inline 桩避免 undefined reference ----
  inline void updateWithOptimizedPoses(const std::vector<Pose6D>& optimized_poses) {
    original_keyframe_poses_ = optimized_poses;
  }

  // 仅声明（实现放 .cpp；避免与 .cpp 重复）
  void updateWithOptimizedPoses(const std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>>& opt);

  // ---- Interfaces implemented elsewhere in project (keep declarations only) ----
  void StateEstimation(StatesGroup& state_propagat);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t,
                      const PointCloudXYZI::Ptr& input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr& trans_cloud);

  void BuildVoxelMap();
  V3F  RGBFromVoxel(const V3D& input_point);
  void UpdateVoxelMap(const std::vector<pointWithVar>& input_points);
  void BuildResidualListOMP(std::vector<pointWithVar>& pv_list, std::vector<PointToPlane>& ptpl_list);
  void build_single_residual(pointWithVar& pv, const VoxelOctoTree* current_octo, const int current_layer,
                             bool& is_success, double& prob, PointToPlane& single_ptpl);
  void pubVoxelMap();

  void mapSliding();
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min);

  // ---------------------- data ----------------------
  VoxelMapConfig config_setting_{};
  int current_frame_id_{0};
  ros::Publisher voxel_map_pub_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*> voxel_map_;

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_{};
  V3D extT_{};
  float build_residual_time{0.f}, ekf_time{0.f};
  float ave_build_residual_time{0.f};
  float ave_ekf_time{0.f};
  int   scan_count{0};
  StatesGroup state_{};
  V3D  position_last_{};
  V3D  last_slide_position{0,0,0};
  geometry_msgs::Quaternion geoQuat_{};

  int feats_down_size_{0};
  int effct_feat_num_{0};
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

  // 关键帧 -> 体素位置列表（如需）
  std::unordered_map<size_t, std::vector<VOXEL_LOCATION>> voxel_keyframe_map_;

  // 保存关键帧位姿（与 PGO 使用的 Pose6D 一致）
  std::vector<Pose6D> original_keyframe_poses_;

private:
  std::mutex voxel_map_mutex_;

  void GetUpdatePlane(const VoxelOctoTree* current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane>& plane_list);
  void pubSinglePlane(visualization_msgs::MarkerArray& plane_pub, const std::string plane_ns,
                      const VoxelPlane& single_plane, const float alpha, const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d& x_vec, const Eigen::Vector3d& y_vec,
                       const Eigen::Vector3d& z_vec, geometry_msgs::Quaternion& q);
  void mapJet(double v, double vmin, double vmax, uint8_t& r, uint8_t& g, uint8_t& b);
};

// shared ptr alias
using VoxelMapManagerPtr = std::shared_ptr<VoxelMapManager>;

#endif // VOXEL_MAP_H_
