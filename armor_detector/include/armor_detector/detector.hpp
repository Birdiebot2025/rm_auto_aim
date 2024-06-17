// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"


// Licensed under the MIT License.

// OpenCV
// #include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
// #include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
// #include <vector>
#include "armor_detector/common.hpp"


namespace rm_auto_aim
{
class Detector
{
public:
  struct LightParams
  {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle
    double max_angle;
  };

  struct ArmorParams
  {
    double min_light_ratio;
    // light pairs distance
    double min_small_center_distance;
    double max_small_center_distance;
    double min_large_center_distance;
    double max_large_center_distance;
    // horizontal angle
    double max_angle;
  };

  // Detector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a);
  Detector(const int & color, const LightParams & l, const ArmorParams & a);

  std::vector<Armor> detect(const cv::Mat & input);

  cv::Mat preprocessImage(const cv::Mat & input);
  std::vector<Light_v8> findLights_v8(std::vector<int> indexes, std::vector<boundingbox> boundingboxs, std::vector<cv::Rect> boxes);
  std::vector<Armor> matchLights_v8(std::vector<rm_auto_aim::v8_cls_confidence> cls, std::vector<Light_v8> lights, std::vector<int> indexes, std::vector<boundingbox> boundingboxs);
  void draw(std::vector<int> indexes, const cv::Mat & input, std::vector<v8_cls_confidence> v8_cls_confidences, std::vector<boundingbox> boundingboxs, std::vector<float> confidences);

  // For debug usage
  cv::Mat getAllNumbersImage();
  void drawResults(cv::Mat & img);
  void loadModel(const std::string &modelXml, const std::string &modelBin);
  int class_id(rm_auto_aim::v8_cls_confidence cls);
  int class_color(rm_auto_aim::v8_cls_confidence cls);
  std::string class_id_number(rm_auto_aim::v8_cls_confidence cls);

  int binary_thres;
  int detect_color;
  LightParams l;
  ArmorParams a;

  std::unique_ptr<NumberClassifier> classifier;

  // Debug msgs
  cv::Mat binary_img;
  auto_aim_interfaces::msg::DebugLights debug_lights;
  auto_aim_interfaces::msg::DebugArmors debug_armors;

private:
  ArmorType isArmor(const Light_v8 & light);
  std::vector<Light_v8> lights_;
  std::vector<Armor> armors_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

};
  
}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
