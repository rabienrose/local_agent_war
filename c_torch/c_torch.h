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

class Net;

namespace torch{
    namespace optim{
        class Adam;
    }
}

class CTorch : public Reference {
    GDCLASS(CTorch, Reference);
protected:
    static void _bind_methods();

private:
    int get_model(String agent_name, int hiddens);
    int get_optim(String agent_name, int hiddens);
    std::map<String, std::shared_ptr<Net>> model_dict;
    std::map<String, std::shared_ptr<torch::optim::Adam>> optim_dict;
    int mini_batch_size=512;
    float learning_rate=0.0001;
    float epoch=3;
    float epsilon=0.2;
    float beta=0.005;

public:
    void set_opti_params(Dictionary setting);
    Dictionary train(String agent_name, Array data, int hiddens);
    void create_model(String agent_name, int hiddens);
    bool load_model(String agent_name, int hiddens);
    PoolVector<float> get_action(const PoolVector<float> &obs, String agent_name, bool b_train);
    Variant test_variant(Variant data);
    CTorch();
};

#endif