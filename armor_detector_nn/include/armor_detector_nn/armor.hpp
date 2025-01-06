// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim {
const int RED = 0;
const int BLUE = 1;

enum class ArmorType { SMALL, LARGE, INVALID };
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

struct Light {
  Light() = default;
  explicit Light(cv::Point2f up, cv::Point2f down) : top(up), bottom(down) {

    center = (top + down) / 2;
    length = cv::norm(top - bottom);

    tilt_angle =
        std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    tilt_angle = tilt_angle / CV_PI * 180;
  }

  int color;
  cv::Point2f top, bottom, center;
  double length;
  float tilt_angle;
};

struct Armor {
  Armor() = default;
  Armor(const Light &l1, const Light &l2) {
    if (l1.center.x < l2.center.x) {
      left_light = l1, right_light = l2;
    } else {
      left_light = l2, right_light = l1;
    }
    center = (left_light.center + right_light.center) / 2;
  }

  // Light pairs part
  Light left_light, right_light;
  cv::Point2f center;
  ArmorType type;

  // Number part
  std::string number;
  float confidence;
  std::string classfication_result;
};

} // namespace rm_auto_aim

#endif // ARMOR_DETECTOR__ARMOR_HPP_
