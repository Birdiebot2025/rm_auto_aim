// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR_NN__DETECTOR_NODE_HPP_
#define ARMOR_DETECTOR_NN__DETECTOR_NODE_HPP_

// ROS
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// STD
#include <memory>
#include <string>
#include <vector>

#include "armor_detector_nn/detector.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include <opencv2/opencv.hpp>

#include "armor_detector_nn/pnp_solver.hpp"

namespace rm_auto_aim {

class ArmorDetectorNode : public rclcpp::Node {
public:
  // for dev
  // std::unique_ptr<cv::VideoCapture> cap;
  // for dev
  ArmorDetectorNode(const rclcpp::NodeOptions &options);

  // Armor Detector
  std::unique_ptr<Detector> detector_;

  // Camera info part
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  cv::Point2f cam_center_;
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
  std::unique_ptr<PnPSolver> pnp_solver_;

  // Image subscrpition
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
  void detectArmorsAsync(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

  std::vector<Armor>
  detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);

  auto_aim_interfaces::msg::Armors armors_msg_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;

  // Visualization marker publisher
  visualization_msgs::msg::Marker armor_marker_;
  visualization_msgs::msg::Marker text_marker_;
  visualization_msgs::msg::MarkerArray marker_array_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;

  void createDebugPublishers();
  void destroyDebugPublishers();
  void publishMarkers();

  std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;

  // Debug information
  bool debug_;
  image_transport::Publisher result_img_pub_;
};

} // namespace rm_auto_aim

#endif // ARMOR_DETECTOR_NN__DETECTOR_NODE_HPP_
