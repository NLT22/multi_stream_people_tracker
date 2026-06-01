/*
 * Custom bounding box parser for YOLOv8 / YOLOv11 pre-NMS output.
 *
 * Supports any number of classes (nc). The output tensor shape is [4+nc, 8400]:
 *   - dim 0: 4+nc  (4 bbox coords + nc class scores)
 *   - dim 1: 8400  anchor predictions
 *
 * Works for both COCO-pretrained (nc=80 → [84,8400]) and fine-tuned
 * single-class models (nc=1 → [5,8400]).  num_classes is inferred from
 * the actual tensor dims at runtime.
 *
 * Reading element [feat][anchor]:
 *   feat 0..3      → cx, cy, w, h  (pixel coords in model input space)
 *   feat 4..4+nc-1 → class scores  (sigmoid already applied by Ultralytics ONNX export)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static const int NUM_ANCHORS = 8400;
static const int BBOX_DIM    = 4;

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo  const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo  const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Find output0 layer
    const NvDsInferLayerInfo* output_layer = nullptr;
    for (const auto& layer : outputLayersInfo) {
        if (std::string(layer.layerName) == "output0") {
            output_layer = &layer;
            break;
        }
    }

    if (!output_layer) {
        std::cerr << "[YoloV8 parser] ERROR: output layer 'output0' not found. "
                     "Available layers:";
        for (const auto& l : outputLayersInfo)
            std::cerr << " '" << l.layerName << "'";
        std::cerr << std::endl;
        return false;
    }

    // Validate dims: expect [4+nc, 8400] for any nc >= 1
    const auto& dims = output_layer->inferDims;
    if (dims.numDims < 2 || dims.d[0] <= BBOX_DIM || dims.d[1] != NUM_ANCHORS) {
        std::cerr << "[YoloV8 parser] ERROR: unexpected output dims ["
                  << dims.d[0] << ", " << dims.d[1] << "]. "
                  << "Expected [4+nc, " << NUM_ANCHORS << "] with nc>=1." << std::endl;
        return false;
    }

    // Infer num_classes from actual tensor dims (supports nc=1..N)
    const int NUM_CLASSES = dims.d[0] - BBOX_DIM;

    const float* data = static_cast<const float*>(output_layer->buffer);
    const float net_w = static_cast<float>(networkInfo.width);
    const float net_h = static_cast<float>(networkInfo.height);

    const float score_threshold = detectionParams.perClassPreclusterThreshold.size() > 0
        ? *std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
                            detectionParams.perClassPreclusterThreshold.end())
        : 0.25f;

    for (int a = 0; a < NUM_ANCHORS; ++a) {
        int   best_class = -1;
        float best_score = score_threshold;

        for (int c = 0; c < NUM_CLASSES; ++c) {
            // Ultralytics ONNX export (simplify=True) applies sigmoid inside the graph.
            // Scores are already in [0, 1] — do NOT apply sigmoid again.
            float score = data[(BBOX_DIM + c) * NUM_ANCHORS + a];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_class < 0) continue;

        // cx,cy,w,h in pixel coords (0..640) — Ultralytics ONNX export decodes coords.
        float cx = data[0 * NUM_ANCHORS + a];
        float cy = data[1 * NUM_ANCHORS + a];
        float bw = data[2 * NUM_ANCHORS + a];
        float bh = data[3 * NUM_ANCHORS + a];

        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        // Clamp to frame
        x1 = std::max(0.0f, std::min(x1, net_w - 1));
        y1 = std::max(0.0f, std::min(y1, net_h - 1));
        x2 = std::max(0.0f, std::min(x2, net_w - 1));
        y2 = std::max(0.0f, std::min(y2, net_h - 1));

        float w = x2 - x1;
        float h = y2 - y1;
        if (w <= 0 || h <= 0) continue;

        NvDsInferParseObjectInfo obj{};
        obj.classId             = static_cast<unsigned int>(best_class);
        obj.detectionConfidence = best_score;
        obj.left                = x1;
        obj.top                 = y1;
        obj.width               = w;
        obj.height              = h;
        objectList.push_back(obj);
    }

    return true;
}
