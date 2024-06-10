// Copyright (c) 2022 ChenJun
#include "armor_detector/detector.hpp"
#include "auto_aim_interfaces/msg/debug_armor.hpp"
#include "auto_aim_interfaces/msg/debug_light.hpp"

#include "armor_detector/common.hpp"
// extern ov::InferRequest infer_request;
namespace rm_auto_aim
{
Detector::Detector(
  const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a)
// : binary_thres(bin_thres), detect_color(color), l(l), a(a)
{
}

std::vector<Armor> Detector::detect(const cv::Mat & input)
{
  // static const std::vector<std::string> class_names = {
  //   "sentry_B", "sentry_N", "sentry_R"
  // };
  static const std::string model_file = "/home/hero/Desktop/yolov8_test/4point_best.onnx";

  try {
        // 获取模型输入节点
        ov::Tensor input_tensor = infer_request_.get_input_tensor();
        // input_tensor = infer_request.get_input_tensor();
        const int64 start = cv::getTickCount();
        // 读取图片并按照模型输入要求进行预处理
        // cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);

        // cv::Mat input ;
        // cv::resize(input_, input, cv::Size(640, 640));
        const float factor = fill_tensor_data_image(input_tensor, input);

        /// 执行推理计算
        infer_request_.infer();

        /// 处理推理计算结果
        // 获得推理结果
        const ov::Tensor output = infer_request_.get_output_tensor();
        const ov::Shape output_shape = output.get_shape();
        const float *output_buffer = output.data<const float>();

        // 解析推理结果
        const int out_rows = output_shape[1]; //获得"output"节点的rows
        const int out_cols = output_shape[2]; //获得"output"节点的cols
        // std::cout << out_rows  << std::endl;
        const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)output_buffer);

        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<Light> lights_all;
        std::vector<boundingbox> boundingboxs;
        std::vector<v8_cls_confidence> v8_cls_confidences;
        // std::vector<std::map<std::string, float>> v8_cls_confidences;

        cv::RotatedRect rect_;
        rect_.center = cv::Point2f(100, 100);
        rect_.size = cv::Size2f(100, 100);
        rect_.angle = 45.f;

        auto lights_ = Light(rect_);
        
        // 输出格式是[84,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是cx, cy, ow, oh, 后面80行是每个类别的置信度
        std::cout << std::endl << std::endl;
        // std::vector<rm_auto_aim::Armor> armors;

        rm_auto_aim::Armor armor;
        rm_auto_aim::boundingbox boundingbox;

        for (int i = 0; i < det_output.cols; ++i) {
            const cv::Mat classes_scores = det_output.col(i).rowRange(4, 6);

            cv::Point class_id_point;
            double score;
            cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

            // 置信度 0～1之间
            if (score > 0.6) {
              rm_auto_aim::v8_cls_confidence v8_cls_confidence;
              // std::map<std::string, float> v8_cls_confidence;
              // v8_cls_confidence["confidence_sentry_B"] = det_output.at<float>(4, i);
              // v8_cls_confidence["confidence_sentry_N"] = det_output.at<float>(5, i);
              // v8_cls_confidence["confidence_sentry_R"] = det_output.at<float>(6, i);
              // std::cout << "----------------------------------" << std::endl;
              // std::cout << det_output.at<float>(4, i) << std::endl;
              // std::cout << det_output.at<float>(5, i) << std::endl;
              // std::cout << det_output.at<float>(6, i) << std::endl;
              // std::cout << "----------------------------------" << std::endl;

              v8_cls_confidence.confidence_sentry_B = det_output.at<float>(4, i);
              v8_cls_confidence.confidence_sentry_N = det_output.at<float>(5, i);
              v8_cls_confidence.confidence_sentry_R = det_output.at<float>(6, i);

              // std::cout << "----------------------------------" << std::endl;
              // std::cout << v8_cls_confidence.confidence_sentry_B << std::endl;
              // std::cout << v8_cls_confidence.confidence_sentry_N << std::endl;
              // std::cout << v8_cls_confidence.confidence_sentry_R << std::endl;
              // std::cout << "----------------------------------" << std::endl;

              float cx = 2.25*det_output.at<float>(0, i);
              float cy = 2.25*det_output.at<float>(1, i);
              float ow = 2.25*det_output.at<float>(2, i);
              float oh = 2.25*det_output.at<float>(3, i);
              // float cx = det_output.at<float>(0, i);
              // float cy = det_output.at<float>(1, i);
              // float ow = det_output.at<float>(2, i);
              // float oh = det_output.at<float>(3, i);

              cv::RotatedRect rect;
              rect.center = cv::Point2f(cx, cy);
              rect.size = cv::Size2f(ow, oh);
              rect.angle = 45.f;

              auto lights = Light(rect);

              boundingbox.p[0].x = 2.25*det_output.at<float>(7, i);
              boundingbox.p[0].y = 2.25*det_output.at<float>(8, i);
              boundingbox.p[3].x = 2.25*det_output.at<float>(9, i);
              boundingbox.p[3].y = 2.25*det_output.at<float>(10, i);

              boundingbox.p[2].x = 2.25*det_output.at<float>(11, i);
              boundingbox.p[2].y = 2.25*det_output.at<float>(12, i);
              boundingbox.p[1].x = 2.25*det_output.at<float>(13, i);
              boundingbox.p[1].y = 2.25*det_output.at<float>(14, i);

              // boundingbox.p[0].x = det_output.at<float>(7, i);
              // boundingbox.p[0].y = det_output.at<float>(8, i);
              // boundingbox.p[3].x = det_output.at<float>(9, i);
              // boundingbox.p[3].y = det_output.at<float>(10, i);

              // boundingbox.p[2].x = det_output.at<float>(11, i);
              // boundingbox.p[2].y = det_output.at<float>(12, i);
              // boundingbox.p[1].x = det_output.at<float>(13, i);
              // boundingbox.p[1].y = det_output.at<float>(14, i);

              lights_all.push_back(lights);
              lights_ = lights;

              cv::Rect box;
              box.x = static_cast<int>((cx - 0.5 * ow) * factor);
              box.y = static_cast<int>((cy - 0.5 * oh) * factor);
              box.width = static_cast<int>(ow * factor);
              box.height = static_cast<int>(oh * factor);

              boundingboxs.push_back(boundingbox);
              boxes.push_back(box);
              // class_ids.push_back(class_id_point.y);
              confidences.push_back(score);
              v8_cls_confidences.push_back(v8_cls_confidence);

            }
        }
        // NMS, 消除具有较低置信度的冗余重叠框
        // std::vector<int> indexes;
        // cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);

        draw(input, v8_cls_confidences, boundingboxs, confidences);

        // std::cout<<indexes.size()<<std::endl;
        std::cout<<boundingboxs.size()<<std::endl;
        std::cout<<confidences.size()<<std::endl;
        // std::cout<<class_ids.size()<<std::endl;
        std::cout<<boxes.size()<<std::endl;
        std::cout<<v8_cls_confidences.size()<<std::endl;
        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << v8_cls_confidences.size() << std::endl;
        cv::putText(input, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        /// 获取程序名称
        // const std::string programName{extractedProgramName(argv[0])};
        // cv::imshow("v8_sentry", input);
        // /// 保存结果图
        // save(programName, image);

        // cv::waitKey(0);
        // cv::destroyAllWindows();

    } catch (const std::exception &e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }
    
  
  // binary_img = preprocessImage(input);
  // lights_ = findLights(input, binary_img);
  // armors_ = matchLights(lights_);
  lights_ = findLights_v8(input, binary_img);
  armors_ = matchLights_v8(lights_);
  
  return armors_;

  // if (!armors_.empty()) {
  //   classifier->extractNumbers(input, armors_);
  //   classifier->classify(armors_);
  // }

  // return armors_;
}
void Detector::loadModel(const std::string &model_file) {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_file);
    printInputAndOutputsInfo(*model); // 打印模型信息 
    compiled_model_ = core.compile_model(model, "AUTO");
    infer_request_ = compiled_model_.create_infer_request();
}

void Detector::draw(const cv::Mat & input, std::vector<v8_cls_confidence> v8_cls_confidences, std::vector<boundingbox> boundingboxs, std::vector<float> confidences){
  static const std::vector<std::string> class_names = {
    "sentry_B", "sentry_N", "sentry_R"
  };
  for (size_t i = 0; i < confidences.size(); i++) {
      std::vector<cv::Point> points;
      for (int j = 0; j < 4; j++) {
          points.emplace_back(boundingboxs[i].p[j].x, boundingboxs[i].p[j].y);
          // std::cout<<points<<std::endl;
      }
      int number = class_id(v8_cls_confidences[i]);
      // auto cls_confidence = std::max_element(v8_cls_confidence.begin(), v8_cls_confidence.end(),
      //   [](const std::pair<std::string, float>& p1, const std::pair<std::string, float>& p2) {
      //       return p1.second < p2.second;
      //   });
      // 绘制四边形
      cv::polylines(input, points, true, cv::Scalar(0, 255, 0), 2, cv::LINE_8);
      // cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
      // 绘制标签
      const std::string label = class_names[number] + ":" + std::to_string(confidences[i]).substr(0, 4);
      // const std::string label = max(v8_cls_confidence, key=dic.get) + ":" + std::to_string(confidences[i]).substr(0, 4);

      const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 20, nullptr);
      const cv::Rect textBox(boundingboxs[i].p[0].x, boundingboxs[i].p[0].y - 15, textSize.width, textSize.height + 5);
      // cv::rectangle(input, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
      cv::putText(input, label, cv::Point(boundingboxs[i].p[0].x, boundingboxs[i].p[0].y), cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 255, 255));
  }

}


// cv::Mat Detector::preprocessImage(const cv::Mat & rgb_img)
// {
//   cv::Mat gray_img;
//   cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);

//   cv::Mat binary_img;
//   cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);

//   return binary_img;
// }

// std::vector<Light> Detector::findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img)
// {
//   using std::vector;
//   vector<vector<cv::Point>> contours;
//   vector<cv::Vec4i> hierarchy;
//   cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//   vector<Light> lights;
//   this->debug_lights.data.clear();

//   for (const auto & contour : contours) {
//     if (contour.size() < 5) continue;

//     auto r_rect = cv::minAreaRect(contour);
//     auto light = Light(r_rect);

//     if (isLight(light)) {
//       auto rect = light.boundingRect();
//       if (  // Avoid assertion failed
//         0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
//         0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
//         int sum_r = 0, sum_b = 0;
//         auto roi = rbg_img(rect);
//         // Iterate through the ROI
//         for (int i = 0; i < roi.rows; i++) {
//           for (int j = 0; j < roi.cols; j++) {
//             if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) {
//               // if point is inside contour
//               sum_r += roi.at<cv::Vec3b>(i, j)[0];
//               sum_b += roi.at<cv::Vec3b>(i, j)[2];
//             }
//           }
//         }
//         // Sum of red pixels > sum of blue pixels ?
//         light.color = sum_r > sum_b ? RED : BLUE;
//         lights.emplace_back(light);
//       }
//     }
//   }

//   return lights;
// }

std::vector<Light> Detector::findLights_v8(const cv::Mat & rbg_img, const cv::Mat & binary_img)
{
  using std::vector;
//   vector<vector<cv::Point>> contours;
//   // vector<cv::Vec4i> hierarchy;
//   // cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

//   vector<Light> lights;
//   // this->debug_lights.data.clear();

//   // for (const auto & contour : contours) {
//   //   if (contour.size() < 5) continue;
// cv::RotatedRect box = 
//     auto r_rect = cv::minAreaRect(contour);
//     auto light = Light(r_rect);

//     if (isLight(light)) {
//       auto rect = light.boundingRect();
//       if (  // Avoid assertion failed
//         0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
//         0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
//         int sum_r = 0, sum_b = 0;
//         auto roi = rbg_img(rect);
//         // Iterate through the ROI
//         for (int i = 0; i < roi.rows; i++) {
//           for (int j = 0; j < roi.cols; j++) {
//             if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) {
//               // if point is inside contour
//               sum_r += roi.at<cv::Vec3b>(i, j)[0];
//               sum_b += roi.at<cv::Vec3b>(i, j)[2];
//             }
//           }
//         }
//         // Sum of red pixels > sum of blue pixels ?
//         light.color = sum_r > sum_b ? RED : BLUE;
//         lights.emplace_back(light);
//       }
//     }
//   // }

  return lights_;
}

int Detector::class_id(rm_auto_aim::v8_cls_confidence cls)
{
  if (cls.confidence_sentry_R >= cls.confidence_sentry_B && cls.confidence_sentry_R >= cls.confidence_sentry_N) {
      return 2;  // red
  } else if (cls.confidence_sentry_B >= cls.confidence_sentry_N && cls.confidence_sentry_B >= cls.confidence_sentry_R) {
      return 0;  // blue
  } else {
      return 1;  // none
  }
}
// bool Detector::isLight(const Light & light)
// {
//   // The ratio of light (short side / long side)
//   float ratio = light.width / light.length;
//   bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

//   bool angle_ok = light.tilt_angle < l.max_angle;

//   bool is_light = ratio_ok && angle_ok;

//   // Fill in debug information
//   auto_aim_interfaces::msg::DebugLight light_data;
//   light_data.center_x = light.center.x;
//   light_data.ratio = ratio;
//   light_data.angle = light.tilt_angle;
//   light_data.is_light = is_light;
//   this->debug_lights.data.emplace_back(light_data);

//   return is_light;
// }

std::vector<Armor> Detector::matchLights_v8(const std::vector<Light> & lights)
{
  std::vector<Armor> armors;
  // this->debug_armors.data.clear();

  // // Loop all the pairing of lights
  // for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
  //   for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
  //     if (light_1->color != detect_color || light_2->color != detect_color) continue;

  //     if (containLight(*light_1, *light_2, lights)) {
  //       continue;
  //     }

  //     auto type = isArmor(*light_1, *light_2);
  //     if (type != ArmorType::INVALID) {
  //       auto armor = Armor(*light_1, *light_2);
  //       armor.type = type;
  //       armors.emplace_back(armor);
  //     }
  //   }
  // }
  return armors;
}
// std::vector<Armor> Detector::matchLights(const std::vector<Light> & lights)
// {
//   std::vector<Armor> armors;
//   this->debug_armors.data.clear();

//   // Loop all the pairing of lights
//   for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
//     for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
//       if (light_1->color != detect_color || light_2->color != detect_color) continue;

//       if (containLight(*light_1, *light_2, lights)) {
//         continue;
//       }

//       auto type = isArmor(*light_1, *light_2);
//       if (type != ArmorType::INVALID) {
//         auto armor = Armor(*light_1, *light_2);
//         armor.type = type;
//         armors.emplace_back(armor);
//       }
//     }
//   }

//   return armors;
// }

// // Check if there is another light in the boundingRect formed by the 2 lights
// bool Detector::containLight(
//   const Light & light_1, const Light & light_2, const std::vector<Light> & lights)
// {
//   auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
//   auto bounding_rect = cv::boundingRect(points);

//   for (const auto & test_light : lights) {
//     if (test_light.center == light_1.center || test_light.center == light_2.center) continue;

//     if (
//       bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
//       bounding_rect.contains(test_light.center)) {
//       return true;
//     }
//   }

//   return false;
// }

// ArmorType Detector::isArmor(const Light & light_1, const Light & light_2)
// {
//   // Ratio of the length of 2 lights (short side / long side)
//   float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
//                                                              : light_2.length / light_1.length;
//   bool light_ratio_ok = light_length_ratio > a.min_light_ratio;

//   // Distance between the center of 2 lights (unit : light length)
//   float avg_light_length = (light_1.length + light_2.length) / 2;
//   float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
//   bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
//                              center_distance < a.max_small_center_distance) ||
//                             (a.min_large_center_distance <= center_distance &&
//                              center_distance < a.max_large_center_distance);

//   // Angle of light center connection
//   cv::Point2f diff = light_1.center - light_2.center;
//   float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
//   bool angle_ok = angle < a.max_angle;

//   bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

//   // Judge armor type
//   ArmorType type;
//   if (is_armor) {
//     type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
//   } else {
//     type = ArmorType::INVALID;
//   }

//   // Fill in debug information
//   auto_aim_interfaces::msg::DebugArmor armor_data;
//   armor_data.type = ARMOR_TYPE_STR[static_cast<int>(type)];
//   armor_data.center_x = (light_1.center.x + light_2.center.x) / 2;
//   armor_data.light_ratio = light_length_ratio;
//   armor_data.center_distance = center_distance;
//   armor_data.angle = angle;
//   this->debug_armors.data.emplace_back(armor_data);

//   return type;
// }

// cv::Mat Detector::getAllNumbersImage()
// {
//   if (armors_.empty()) {
//     return cv::Mat(cv::Size(20, 28), CV_8UC1);
//   } else {
//     std::vector<cv::Mat> number_imgs;
//     number_imgs.reserve(armors_.size());
//     for (auto & armor : armors_) {
//       number_imgs.emplace_back(armor.number_img);
//     }
//     cv::Mat all_num_img;
//     cv::vconcat(number_imgs, all_num_img);
//     return all_num_img;
//   }
// }

// void Detector::drawResults(cv::Mat & img)
// {
//   // Draw Lights
//   for (const auto & light : lights_) {
//     cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
//     cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
//     auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
//     cv::line(img, light.top, light.bottom, line_color, 1);
//   }

//   // Draw armors
//   for (const auto & armor : armors_) {
//     cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
//     cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
//   }

//   // Show numbers and confidence
//   for (const auto & armor : armors_) {
//     cv::putText(
//       img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
//       cv::Scalar(0, 255, 255), 2);
//   }
// }

}  // namespace rm_auto_aim
