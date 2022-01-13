/* summator.cpp */
// #include <android/log.h>
#include "c_torch.h"
#include "core/os/os.h"
#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(8, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc_o1 = register_module("fc_o1", torch::nn::Linear(32, 3));
        fc_o2 = register_module("fc_o2", torch::nn::Linear(32, 3));
    }
    std::vector<torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        torch::Tensor out1 = torch::log_softmax(fc_o1->forward(x), /*dim=*/1);
        torch::Tensor out2 = torch::log_softmax(fc_o2->forward(x), /*dim=*/1);
        std::vector<torch::Tensor> outs;
        outs.push_back(out1);
        outs.push_back(out2);
        return outs;
    }
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc_o1{ nullptr }, fc_o2{ nullptr };
};

void CTorch::create_model() {
    auto net = std::make_shared<Net>();
    String usr_path = OS::get_singleton()->get_user_data_dir();
    usr_path=usr_path+"/test.pt";
    std::wstring ws = usr_path.c_str();
    std::string s( ws.begin(), ws.end() );
    torch::save(net, s.c_str());
}

void CTorch::test_train() {
    auto net = std::make_shared<Net>();
    auto data_loader = torch::data::make_data_loader(torch::data::datasets::MNIST("/data/data/org.godotengine.agentwar/files/data").map(torch::data::transforms::Stack<>()),64);
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        for (auto &batch : *data_loader) {
            optimizer.zero_grad();
            // torch::Tensor prediction = net->forward(batch.data);
            // torch::Tensor loss = torch::nll_loss(prediction, batch.target);

            // loss.backward();
            // optimizer.step();
            // if (++batch_index % 100 == 0) {
            //     print_line(vformat("Epoch: %d | Batch: %d | Loss: %d",epoch,batch_index,loss.item<float>()));
            //     torch::save(net, "/data/data/org.godotengine.agentwar/files/net.pt");
            // }
        }
    }
}

PoolVector<int> CTorch::get_action(const PoolVector<float> &obs) {

    auto obs1_a = obs_tensor[0].accessor<float,2>();
    for(int i = 0; i < 2; i++) {
        obs1_a[0][i]=obs[i];
    }
    auto obs2_a = obs_tensor[1].accessor<float,3>();
    for(int i = 0; i < 20; i++) {
        for(int j = 0; j < 4; j++) {
            obs2_a[0][i][j]=obs[2+i*4+j];
        }
    }
    // std::cout<<obs_tensor[0]<<std::endl;
    // std::cout<<obs_tensor[1]<<std::endl;
    at::Tensor output = module.forward(inputs).toTuple()->elements()[2].toTensor();
    auto out_acc = output.accessor<long,2>();
    for (int i=0; i<outs.size(); i++){
        outs.set(i, out_acc[0][i]);
    }
    return outs;
}

// // Some common guards for inference-only custom mobile LibTorch.
// struct MobileCallGuard {
//   // AutoGrad is disabled for mobile by default.
//   torch::autograd::AutoGradMode no_autograd_guard{false};
//   // VariableType dispatch is not included in default mobile build. We need set
//   // this guard globally to avoid dispatch error (only for dynamic dispatch).
//   // Thanks to the unification of Variable class and Tensor class it's no longer
//   // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
//   // always set NonVariableTypeMode for inference only use case.
//   torch::AutoNonVariableTypeMode non_var_guard{true};
//   // Disable graph optimizer to ensure list of unused ops are not changed for
//   // custom mobile build.
//   torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
// };

// torch::jit::script::Module loadModel(const std::string& path) {
//   MobileCallGuard guard;
//   auto module = torch::jit::load(path);
//   module.eval();
//   return module;
// }

void debug_tensor(at::IValue tensor, std::string tensor_name) {
    std::ostringstream stream;
    stream << tensor;
    print_line(vformat("%s: %s", tensor_name.c_str(), stream.str().c_str()));
}

// debug_tensor(mask, "mask");
// __android_log_print(ANDROID_LOG_VERBOSE, "chamo", "debug_tensor  %s", debug_tensor(mask).c_str());
// if (OS::get_singleton()->get_name()=="X11"){
//     __android_log_print(ANDROID_LOG_VERBOSE, "chamo", "model path  %s", s.c_str());
// }else{
//     std::cout<<"model path: "<<s<<std::endl;
// }

int CTorch::load_model(String model_path) {

    torch::jit::IValue obs = obs_tensor;
    torch::jit::IValue mask = torch::ones({1,4});
    torch::jit::IValue mem = torch::ones({1,1,0});

    inputs.push_back(obs);
    inputs.push_back(mask);
    inputs.push_back(mem);
    outs.resize(4);


    std::wstring ws = model_path.c_str();
    std::string s(ws.begin(), ws.end());
    try {

        print_line(vformat("model path: %s", s.c_str()));
        // auto model = torch::jit::load(s.c_str());
        auto model = torch::jit::_load_for_mobile(s.c_str());
        std::vector<at::IValue> inputs;
        std::vector<at::Tensor> obss;
        obss.push_back(at::ones({ 1, 2 }));
        obss.push_back(at::ones({ 1, 20, 4 }));
        inputs.push_back(obss);
        inputs.push_back(at::ones({ 1, 8 }));
        inputs.push_back(at::ones({ 1, 1, 0 }));
        at::IValue output;
        for (int i = 0; i < 50; i++) {
            int t1 = OS::get_singleton()->get_ticks_msec();
            output = model.forward(inputs);
            int t2 = OS::get_singleton()->get_ticks_msec();
            print_line(vformat("elaps: %d", t2 - t1));
        }

        debug_tensor(output, "out");

        // for (auto& iv: ret) {
        //     debug_tensor(iv, "chamo");
        // }
    } catch (const c10::Error &e) {

        print_error(vformat("error loading the model: %s", e.what()));
        return -1;
    }
    print_line("model load succ");
    return 0;
}

void CTorch::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "model_path"), &CTorch::load_model);
    ClassDB::bind_method(D_METHOD("get_action", "obs1"), &CTorch::get_action);
    ClassDB::bind_method(D_METHOD("test_train"), &CTorch::test_train);
    ClassDB::bind_method(D_METHOD("create_model"), &CTorch::create_model);
}

CTorch::CTorch() {
}