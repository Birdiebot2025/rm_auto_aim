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

struct Light : public cv::RotatedRect
{
  Light() = default;
  explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
  {
    cv::Point2f p[4];
    box.points(p);
    std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
    top = (p[0] + p[1]) / 2;
    bottom = (p[2] + p[3]) / 2;

    length = cv::norm(top - bottom);
    width = cv::norm(p[0] - p[1]);

    // tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    // tilt_angle = tilt_angle / CV_PI * 180;
    tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    tilt_angle = tilt_angle / CV_PI * 180;
  }

  int color;
  cv::Point2f top, bottom;
  double length;
  double width;
  float tilt_angle;
};

struct Light_v8 : public cv::RotatedRect
{
  Light_v8() = default;
  explicit Light_v8(cv::RotatedRect box) : cv::RotatedRect(box)
  {
    cv::Point2f p[4];
    box.points(p);
    std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
    top = (p[0] + p[1]) / 2;
    bottom = (p[2] + p[3]) / 2;

    // tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    // tilt_angle = tilt_angle / CV_PI * 180;
    tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
    tilt_angle = tilt_angle / CV_PI * 180;
  }

  int color;
  cv::Point2f top, bottom;
  float tilt_angle;
};

struct boundingbox
{
  struct {
      float x;
      float y;
  } p[4];
};


struct Armor
{
  Armor() = default;
  Armor(const Light & l1, const Light & l2)
  {
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
  // cv::Mat number_img;
  std::string number;
  float confidence;
  std::string classfication_result;
};

// struct v8_img_data
// {
//   std::vector<int> indexes;
//   rm_auto_aim::boundingbox boundingbox;
//   int class_id;
//   float confidence;
// };

// struct v8_cls_confidence
// {
//   // float confidence_sentry_B;
//   // float confidence_sentry_N;
//   // float confidence_sentry_R;
//   float confidence_B1;
//   float confidence_B2;
//   float confidence_B3;
//   float confidence_B4;
//   float confidence_B5;
//   float confidence_BO;
//   float confidence_BS;
//   float confidence_R1;
//   float confidence_R2;
//   float confidence_R3;
//   float confidence_R4;
//   float confidence_R5;
//   float confidence_RO;
//   float confidence_RS;
// };

struct v8_cls_confidence {
  std::array<float, 14> confidence_cls;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
