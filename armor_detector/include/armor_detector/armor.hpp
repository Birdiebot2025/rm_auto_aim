// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>

// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{
const int RED = 0;
const int BLUE = 1;

enum class ArmorType { SMALL, LARGE, INVALID };
const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};

struct boundingbox
{
  struct {
      float x;
      float y;
  } p[4];
};

struct Light_v8 : public cv::Rect
{
  Light_v8() = default;
  explicit Light_v8(const boundingbox & bbox) 
  {
    cv::Point2f p[4];
    for (int i = 0; i < 4; i++) {
      p[i].x = bbox.p[i].x;
      p[i].y = bbox.p[i].y;
    }
    std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });

    left_top = p[0];
    right_bottom = p[3];
    right_top = p[1];
    left_bottom = p[2];

    left_center = (left_top + left_bottom)/2;
    right_center = (right_top + right_bottom)/2;

    left_length = left_bottom.y - left_top.y;
    right_length = right_bottom.y - right_top.y;
    color = 0;
    // tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    // tilt_angle = tilt_angle / CV_PI * 180;
  }

  int color;
  cv::Point2f left_top, left_bottom, right_top, right_bottom;
  cv::Point2f left_center, right_center;
  double left_length, right_length;
  // double left_width, right_width;
  // float tilt_angle;
};

struct Armor
{
  Armor() = default;
  Armor(const boundingbox & bbox)
  {
    for (int i = 0; i < 4; i++) {
      light_points.p[i].x = bbox.p[i].x;
      light_points.p[i].y = bbox.p[i].y;
    }
    center.x =  ((light_points.p[1].x - light_points.p[0].x) + (light_points.p[2].x - light_points.p[3].x))/2;
    center.y =  ((light_points.p[2].y - light_points.p[1].y) + (light_points.p[3].y - light_points.p[0].y))/2;

  }
  boundingbox light_points;
  cv::Point2f center;
  ArmorType type;
  std::string number;
  float confidence;
  std::string classfication_result;
};

struct v8_cls_confidence {
  std::array<float, 14> confidence_cls;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
