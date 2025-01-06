#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "armor_detector_nn/armor.hpp"

namespace rm_auto_aim {

class Detector {
public:
  struct Config {
    float confThreshold;
    float nmsThreshold;
    float scoreThreshold;
    int inpWidth;
    int inpHeight;
    std::string xml_path;
  };
  Detector(Config &config);

  std::vector<Armor> detect(const cv::Mat &input);
  void preprocessImage(const cv::Mat &frame);
  std::vector<Armor> postprocess(float *detections, ov::Shape &output_shape);
  ArmorType isArmor(const Light &light_1, const Light &light_2);
  float min_large_center_distance;
  void drawResults(cv::Mat &img);

  int detect_color;
  std::vector<long int> ignore_classes_;
  ov::InferRequest infer_request;


private:
  Config config_;
  float rx; // the width ratio of original image and resized image
  float ry; // the height ratio of original image and resized image
  int dx;
  int dy;
  // frame size
  int frame_w;
  int frame_h;
  ov::Tensor input_tensor;
  ov::CompiledModel compiled_model;
  std::vector<Armor> armors_;
};
} // namespace rm_auto_aim

#endif // ARMOR_DETECTOR__DETECTOR_HPP_