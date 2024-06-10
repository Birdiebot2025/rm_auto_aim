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

  Detector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a);

  std::vector<Armor> detect(const cv::Mat & input);

  cv::Mat preprocessImage(const cv::Mat & input);
  // std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img);
  // std::vector<Armor> matchLights(const std::vector<Light> & lights);
  std::vector<Light> findLights_v8(const cv::Mat & rbg_img, const cv::Mat & binary_img);
  std::vector<Armor> matchLights_v8(const std::vector<Light> & lights);
  void draw(std::vector<int> indexes, const cv::Mat & input, std::vector<v8_cls_confidence> v8_cls_confidences, std::vector<boundingbox> boundingboxs, std::vector<float> confidences);

  // For debug usage
  cv::Mat getAllNumbersImage();
  void drawResults(cv::Mat & img);
  void loadModel(const std::string &model_file);
  int class_id(rm_auto_aim::v8_cls_confidence cls);

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
  bool isLight(const Light & possible_light);
  bool containLight(
    const Light & light_1, const Light & light_2, const std::vector<Light> & lights);
  ArmorType isArmor(const Light & light_1, const Light & light_2);

  std::vector<Light> lights_;
  std::vector<Armor> armors_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

};
  
}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
