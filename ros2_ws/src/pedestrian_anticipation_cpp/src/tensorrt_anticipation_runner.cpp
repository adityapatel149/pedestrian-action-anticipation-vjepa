#include "pedestrian_anticipation_cpp/tensorrt_anticipation_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace nvinfer1;

namespace {
    template <typename T>
    std::vector<int64_t> dimsToShape(const T& dims) {
        std::vector<int64_t> out;
        for (int i = 0; i < dims.nbDims; ++i)
            out.push_back(dims.d[i]);
        return out;
    }
}

TensorRTAnticipationRunner::DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
{
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
}

TensorRTAnticipationRunner::DeviceBuffer&
TensorRTAnticipationRunner::DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
    if (this != &other) {
        if (ptr) cudaFree(ptr);
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    return *this;
}
void TensorRTAnticipationRunner::Logger::log(Severity s, const char* msg) noexcept {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
}

TensorRTAnticipationRunner::DeviceBuffer::~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
}

void TensorRTAnticipationRunner::DeviceBuffer::allocate(size_t n) {
    if (n <= bytes && ptr) return;
    if (ptr) cudaFree(ptr);
    checkCuda(cudaMalloc(&ptr, n), "cudaMalloc");
    bytes = n;
}

TensorRTAnticipationRunner::TensorRTAnticipationRunner(
    const std::string& enc,
    const std::string& cls,
    int device_id)
{
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
    checkCuda(cudaStreamCreate(&stream_), "stream");

    runtime_.reset(createInferRuntime(logger_));
    encoder_ = loadEngine(enc);
    classifier_ = loadEngine(cls);
}

TensorRTAnticipationRunner::~TensorRTAnticipationRunner() {
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<char> TensorRTAnticipationRunner::readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("open failed");
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> d(sz);
    f.read(d.data(), sz);
    return d;
}

TensorRTAnticipationRunner::EngineContext
TensorRTAnticipationRunner::loadEngine(const std::string& path) {
    EngineContext ec;
    auto blob = readFile(path);

    ec.engine.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    ec.context.reset(ec.engine->createExecutionContext());

    ec.nb_bindings = ec.engine->getNbIOTensors();
    ec.device_buffers.resize(ec.nb_bindings);

    return ec;
}

int TensorRTAnticipationRunner::findBindingIndex(const ICudaEngine& e, const std::string& name) {
    for (int i = 0; i < e.getNbIOTensors(); ++i)
        if (name == e.getIOTensorName(i)) return i;
    throw std::runtime_error("tensor not found: " + name);
}

size_t TensorRTAnticipationRunner::volume(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) {
            throw std::runtime_error("volume() received unresolved or invalid dims");
        }
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

void TensorRTAnticipationRunner::checkCuda(cudaError_t s, const char* w) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(w) + cudaGetErrorString(s));
}

std::vector<float> TensorRTAnticipationRunner::softmax2(const std::vector<float>& x) {
    std::vector<float> o(x.size());
    for (size_t i = 0; i < x.size(); i += 2) {
        float m = std::max(x[i], x[i + 1]);
        float a = std::exp(x[i] - m), b = std::exp(x[i + 1] - m);
        float s = a + b;
        o[i] = a / s; o[i + 1] = b / s;
    }
    return o;
}

std::vector<float> TensorRTAnticipationRunner::runEncoder(
    const std::vector<float>& clip,
    const std::vector<int64_t>& shape,
    float t,
    std::vector<int64_t>& out_shape)
{
    auto& ctx = *encoder_.context;

    Dims clip_d{}; clip_d.nbDims = shape.size();
    for (int i = 0; i < clip_d.nbDims; ++i) clip_d.d[i] = shape[i];

    Dims ant_d{}; ant_d.nbDims = 1; ant_d.d[0] = 1;

    ctx.setInputShape(enc_clip_name_.c_str(), clip_d);
    ctx.setInputShape(enc_ant_name_.c_str(), ant_d);

    auto out_d = ctx.getTensorShape(enc_out_name_.c_str());

    size_t clip_bytes = clip.size() * 4;
    size_t ant_bytes = 4;
    size_t out_bytes = volume(out_d) * 4;

    int clip_i = findBindingIndex(*encoder_.engine, enc_clip_name_);
    int ant_i = findBindingIndex(*encoder_.engine, enc_ant_name_);
    int out_i = findBindingIndex(*encoder_.engine, enc_out_name_);

    encoder_.device_buffers[clip_i].allocate(clip_bytes);
    encoder_.device_buffers[ant_i].allocate(ant_bytes);
    encoder_.device_buffers[out_i].allocate(out_bytes);

    float tval = t;

    cudaMemcpyAsync(encoder_.device_buffers[clip_i].ptr, clip.data(), clip_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(encoder_.device_buffers[ant_i].ptr, &tval, ant_bytes, cudaMemcpyHostToDevice, stream_);

    ctx.setTensorAddress(enc_clip_name_.c_str(), encoder_.device_buffers[clip_i].ptr);
    ctx.setTensorAddress(enc_ant_name_.c_str(), encoder_.device_buffers[ant_i].ptr);
    ctx.setTensorAddress(enc_out_name_.c_str(), encoder_.device_buffers[out_i].ptr);

    if (!ctx.enqueueV3(stream_)) throw std::runtime_error("enc failed");

    std::vector<float> out(volume(out_d));
    cudaMemcpyAsync(out.data(), encoder_.device_buffers[out_i].ptr, out_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    out_shape = dimsToShape(out_d);
    return out;
}

std::vector<float> TensorRTAnticipationRunner::runClassifier(
    const std::vector<float>& feat,
    const std::vector<int64_t>& feat_shape,
    const std::vector<float>& bbox,
    const std::vector<int64_t>& bbox_shape)
{
    auto& ctx = *classifier_.context;

    Dims fd{}, bd{};
    fd.nbDims = static_cast<int>(feat_shape.size());
    bd.nbDims = static_cast<int>(bbox_shape.size());

    for (int i = 0; i < fd.nbDims; ++i) fd.d[i] = static_cast<int>(feat_shape[i]);
    for (int i = 0; i < bd.nbDims; ++i) bd.d[i] = static_cast<int>(bbox_shape[i]);

    if (!ctx.setInputShape(cls_feat_name_.c_str(), fd)) {
        throw std::runtime_error("setInputShape failed for classifier features");
    }
    if (!ctx.setInputShape(cls_bbox_name_.c_str(), bd)) {
        throw std::runtime_error("setInputShape failed for classifier bboxes");
    }

    if (!ctx.allInputDimensionsSpecified()) {
        throw std::runtime_error("classifier input dimensions not fully specified");
    }

#if NV_TENSORRT_MAJOR >= 10
    int32_t n_unresolved = ctx.inferShapes(0, nullptr);
    if (n_unresolved < 0) {
        throw std::runtime_error("classifier inferShapes failed");
    }
#endif

    auto cross_d = ctx.getTensorShape(cls_cross_name_.c_str());
    auto action_d = ctx.getTensorShape(cls_action_name_.c_str());
    auto intersection_d = ctx.getTensorShape(cls_intersection_name_.c_str());
    auto signalized_d = ctx.getTensorShape(cls_signalized_name_.c_str());

    auto ensureResolved = [](const Dims& d, const std::string& name) {
        if (d.nbDims <= 0) {
            throw std::runtime_error("invalid dims for output tensor: " + name);
        }
        for (int i = 0; i < d.nbDims; ++i) {
            if (d.d[i] <= 0) {
                throw std::runtime_error("unresolved output dim for tensor: " + name);
            }
        }
        };

    ensureResolved(cross_d, cls_cross_name_);
    ensureResolved(action_d, cls_action_name_);
    ensureResolved(intersection_d, cls_intersection_name_);
    ensureResolved(signalized_d, cls_signalized_name_);

    int fi = findBindingIndex(*classifier_.engine, cls_feat_name_);
    int bi = findBindingIndex(*classifier_.engine, cls_bbox_name_);
    int coi = findBindingIndex(*classifier_.engine, cls_cross_name_);
    int aoi = findBindingIndex(*classifier_.engine, cls_action_name_);
    int ioi = findBindingIndex(*classifier_.engine, cls_intersection_name_);
    int soi = findBindingIndex(*classifier_.engine, cls_signalized_name_);

    const size_t feat_bytes = feat.size() * sizeof(float);
    const size_t bbox_bytes = bbox.size() * sizeof(float);

    const size_t cross_bytes = volume(cross_d) * sizeof(float);
    const size_t action_bytes = volume(action_d) * sizeof(float);
    const size_t intersection_bytes = volume(intersection_d) * sizeof(float);
    const size_t signalized_bytes = volume(signalized_d) * sizeof(float);

    classifier_.device_buffers[fi].allocate(feat_bytes);
    classifier_.device_buffers[bi].allocate(bbox_bytes);
    classifier_.device_buffers[coi].allocate(cross_bytes);
    classifier_.device_buffers[aoi].allocate(action_bytes);
    classifier_.device_buffers[ioi].allocate(intersection_bytes);
    classifier_.device_buffers[soi].allocate(signalized_bytes);

    checkCuda(cudaMemcpyAsync(
        classifier_.device_buffers[fi].ptr,
        feat.data(),
        feat_bytes,
        cudaMemcpyHostToDevice,
        stream_), "cudaMemcpyAsync classifier features");

    checkCuda(cudaMemcpyAsync(
        classifier_.device_buffers[bi].ptr,
        bbox.data(),
        bbox_bytes,
        cudaMemcpyHostToDevice,
        stream_), "cudaMemcpyAsync classifier bboxes");

    if (!ctx.setTensorAddress(cls_feat_name_.c_str(), classifier_.device_buffers[fi].ptr)) {
        throw std::runtime_error("setTensorAddress failed for classifier features");
    }
    if (!ctx.setTensorAddress(cls_bbox_name_.c_str(), classifier_.device_buffers[bi].ptr)) {
        throw std::runtime_error("setTensorAddress failed for classifier bboxes");
    }
    if (!ctx.setTensorAddress(cls_cross_name_.c_str(), classifier_.device_buffers[coi].ptr)) {
        throw std::runtime_error("setTensorAddress failed for classifier cross");
    }
    if (!ctx.setTensorAddress(cls_action_name_.c_str(), classifier_.device_buffers[aoi].ptr)) {
        throw std::runtime_error("setTensorAddress failed for classifier action");
    }
    if (!ctx.setTensorAddress(cls_intersection_name_.c_str(), classifier_.device_buffers[ioi].ptr)) {
        throw std::runtime_error("setTensorAddress failed for classifier intersection");
    }
    if (!ctx.setTensorAddress(cls_signalized_name_.c_str(), classifier_.device_buffers[soi].ptr)) {
        throw std::runtime_error("setTensorAddress failed for classifier signalized");
    }

    if (!ctx.enqueueV3(stream_)) {
        throw std::runtime_error("cls failed");
    }

    std::vector<float> cross_out(volume(cross_d));
    checkCuda(cudaMemcpyAsync(
        cross_out.data(),
        classifier_.device_buffers[coi].ptr,
        cross_bytes,
        cudaMemcpyDeviceToHost,
        stream_), "cudaMemcpyAsync classifier cross output");

    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize classifier");
    return cross_out;
}


std::vector<std::pair<int, float>> TensorRTAnticipationRunner::predict(
    const std::vector<float>& clip,
    const std::vector<int64_t>& clip_shape,
    const std::vector<int>& ids,
    const std::vector<float>& bbox,
    const std::vector<int64_t>& bbox_shape,
    float t)
{
    if (ids.empty()) return {};

    std::vector<int64_t> feat_shape;
    auto feat = runEncoder(clip, clip_shape, t, feat_shape);

    size_t one = feat.size();
    std::vector<float> rep(one * ids.size());

    for (size_t i = 0; i < ids.size(); ++i)
        memcpy(rep.data() + i * one, feat.data(), one * 4);

    feat_shape[0] = ids.size();

    auto logits = runClassifier(rep, feat_shape, bbox, bbox_shape);
    auto probs = softmax2(logits);

    std::vector<std::pair<int, float>> out;
    for (size_t i = 0; i < ids.size(); ++i)
        out.push_back({ ids[i], probs[i * 2 + 1] });

    return out;
}