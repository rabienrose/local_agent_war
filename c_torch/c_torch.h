#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"
#include <torch/script.h>
#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/script.h"
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>

class CTorch : public Reference {
    GDCLASS(CTorch, Reference);
    torch::jit::mobile::Module module;
    std::vector<at::Tensor> obs_tensor;
    std::vector<torch::jit::IValue> inputs;
    PoolVector<int> outs;
protected:
    static void _bind_methods();

public:
    PoolVector<int> get_action(const PoolVector<float> &obs);
    void test_train();
    void create_model();
    int load_model(String model_path);

    CTorch();
};

#endif // SUMMATOR_H