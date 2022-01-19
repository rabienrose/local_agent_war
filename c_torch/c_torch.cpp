#include "c_torch.h"
#include "core/os/os.h"
#include <torch/torch.h>
#include <iostream>
#include <math.h> 
using namespace std::chrono;

int model_input_size=10;

struct Net : torch::nn::Module {
    Net(int hiddens) {
        fc1 = register_module("fc1", torch::nn::Linear(model_input_size, hiddens));
        fc2 = register_module("fc2", torch::nn::Linear(hiddens, hiddens));
        fc_o1 = register_module("fc_o1", torch::nn::Linear(hiddens, 3));
        fc_o2 = register_module("fc_o2", torch::nn::Linear(hiddens, 3));
        value = register_module("value", torch::nn::Linear(hiddens, 1));
    }
    std::vector<torch::Tensor> forward(torch::Tensor x, bool b_value) {
        x = torch::leaky_relu(fc1->forward(x));
        x = torch::leaky_relu(fc2->forward(x));
        torch::Tensor prob1 = torch::softmax(fc_o1->forward(x), /*dim=*/1);
        torch::Tensor prob2 = torch::softmax(fc_o2->forward(x), /*dim=*/1);
        torch::Tensor act1 = torch::multinomial(prob1,1);
        torch::Tensor act2 = torch::multinomial(prob2,1);
        std::vector<torch::Tensor> outs;
        outs.push_back(act1);
        outs.push_back(act2);
        if (b_value){
            torch::Tensor out_value = value->forward(x);
            outs.push_back(out_value);
            outs.push_back(prob1);
            outs.push_back(prob2);
        }
        return outs;
    }
    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear fc_o1{ nullptr };
    torch::nn::Linear fc_o2{ nullptr };
    torch::nn::Linear value{ nullptr };
};

void debug_tensor(at::IValue tensor, std::string tensor_name) {
    std::ostringstream stream;
    stream << tensor;
    print_line(vformat("%s: %s", tensor_name.c_str(), stream.str().c_str()));
}

std::string get_model_full_path(String agent_name, bool b_optim){
    String usr_path = OS::get_singleton()->get_user_data_dir();
    String suffix="_model";
    if (b_optim){
        suffix="_optim";
    }
    usr_path=usr_path+"/"+agent_name+suffix+".pt";
    std::wstring ws = usr_path.c_str();
    std::string s( ws.begin(), ws.end());
    return s;
}

template<typename T>
std::vector<T> convert_tensor_2_vector_1d(torch::Tensor vals){
    std::vector<T> out;
    int size=vals.size(1);
    out.resize(size);
    auto val_acc = vals.accessor<T,2>();
    for (int i=0; i<size; i++){
        out[i]=val_acc[0][i];
    }
    return out;
}

torch::Tensor conver_array_2_tensor(Array array){
    int col=array.size();
    int row=Array(array[0]).size();
    torch::Tensor out = torch::zeros({col, row});
    auto out_acc = out.accessor<float,2>();
    for (int j=0; j<col; j++){
        Array row_data=Array(array[j]);
        for (int i=0; i<row; i++){
            out_acc[j][i]=row_data[i];
        }
    }
    return out;
}

torch::Tensor get_tensor_by_type(Array data, int start_ind, int end_ind, String type){
    int data_len=end_ind-start_ind;
    Array temp_type_data = Array(Dictionary(data[0])[type]);
    int data_width = temp_type_data.size();
    if (data_width==0){
        data_width=1;
    }
    torch::Tensor out =torch::zeros({data_len, data_width});
    auto out_acc = out.accessor<float,2>();
    for (int j=0; j<data_len; j++){
        if (data_width==1){
            out_acc[j][0]=(float)Dictionary(data[j])[type];
        }else{
            Array row_data=Array(Dictionary(data[j])[type]);
            for (int i=0; i<data_width; i++){
                out_acc[j][i]=(float)row_data[i];
            }
        }
    }
    return out;
}

int CTorch::get_model(String agent_name, int hiddens){
    if (model_dict.count(agent_name)){
        return 0;
    }else{
        std::string path_f = get_model_full_path(agent_name, false);
        if (FileAccess::exists(path_f.c_str())){
            std::shared_ptr<Net> net = std::make_shared<Net>(hiddens);
            torch::load(net, path_f.c_str());
            model_dict[agent_name]=net;
            return 1;
        }else{
            create_model(agent_name, hiddens);
            return 2;
        }
    }
}

int CTorch::get_optim(String agent_name, int hiddens){
    if (optim_dict.count(agent_name)){
        return 0;
    }else{
        get_model(agent_name, hiddens);
        std::string path_f = get_model_full_path(agent_name, true);
        std::shared_ptr<torch::optim::Adam> optim_ptr=std::make_shared<torch::optim::Adam>(model_dict[agent_name]->parameters());
        optim_dict[agent_name]=optim_ptr;
        if (FileAccess::exists(path_f.c_str())){
            torch::load(*optim_ptr, path_f.c_str());
            return 1;
        }else{
            return 2;
        }
    }
}

void CTorch::create_model(String agent_name, int hiddens) {
    std::shared_ptr<Net> net = std::make_shared<Net>(hiddens);
    model_dict[agent_name]=net;
    std::string full_path = get_model_full_path(agent_name, false);
    torch::save(net, full_path.c_str());
}

bool CTorch::load_model(String agent_name, int hiddens) {
    std::shared_ptr<Net> net = std::make_shared<Net>(hiddens);
    std::string full_path = get_model_full_path(agent_name, false);
    torch::load(net, full_path.c_str());
    model_dict[agent_name]=net;
    return true;
}

Variant CTorch::test_variant(Variant data){
    print_line("=====================");
    print_line(data.get("sss"));
    data.set("sss",999);
    Dictionary var;
    var["asf"]=11;
    return var;
}

PoolVector<float> CTorch::get_action(const PoolVector<float> &obs, String agent_name, bool b_train) {
    PoolVector<float> outs;
    if (!b_train){
        outs.resize(2);
    }else{
        outs.resize(9);
    }
    
    torch::Tensor obs_at = torch::ones({1,obs.size()});
    auto obs_acc = obs_at.accessor<float,2>();
    for(int i = 0; i < obs.size(); i++) {
        obs_acc[0][i]=obs[i];
    }
    std::shared_ptr<Net> net;
    if (model_dict.count(agent_name)){
        net=model_dict[agent_name];
        std::vector<torch::Tensor> outs_at = net->forward(obs_at, b_train);
        PoolVector<float>::Write w = outs.write();
        w[0]=convert_tensor_2_vector_1d<long>(outs_at[0])[0];
        w[1]=convert_tensor_2_vector_1d<long>(outs_at[1])[0];
        if (b_train){
            // debug_tensor(outs_at[2], "value");
            w[2]=convert_tensor_2_vector_1d<float>(outs_at[2])[0];
            std::vector<float> probs1=convert_tensor_2_vector_1d<float>(outs_at[3]);
            std::vector<float> probs2=convert_tensor_2_vector_1d<float>(outs_at[4]);
            w[3]=probs1[0];
            w[4]=probs1[1];
            w[5]=probs1[2];
            w[6]=probs2[0];
            w[7]=probs2[1];
            w[8]=probs2[2];
        }
    }
    return outs;
}

Dictionary CTorch::train(String agent_name, Array data, int hiddens) {
    float EPSILON=1e-7;
    get_optim(agent_name, hiddens);
    auto optim_ptr = optim_dict[agent_name];
    auto options = static_cast<torch::optim::AdamOptions&>(optim_ptr->defaults());
    options.lr(learning_rate);
    float avg_policy_loss=0;
    float avg_value_loss=0;
    float avg_entropy_loss=0;
    int total_step_num=data.size();
    int mini_batch_count=floor(total_step_num/mini_batch_size);
    int total_update=epoch*mini_batch_count;
    float f1=0;
    float f2=0;
    float f3=0;
    float f4=0;
    for (int n=0; n<epoch; n++){
        data.shuffle();
        for (int i=0; i<mini_batch_count; i++){
            int start_ind=i*mini_batch_size;
            int end_ind=(i+1)*mini_batch_size;
            auto t1 = high_resolution_clock::now();
            torch::Tensor obss=get_tensor_by_type(data, start_ind, end_ind, "obs");
            torch::Tensor actions=get_tensor_by_type(data, start_ind, end_ind, "action");
            torch::Tensor advantages=get_tensor_by_type(data, start_ind, end_ind, "advantage");
            torch::Tensor old_prob=get_tensor_by_type(data, start_ind, end_ind, "prob");
            torch::Tensor old_values=get_tensor_by_type(data, start_ind, end_ind, "value");
            torch::Tensor returns=get_tensor_by_type(data, start_ind, end_ind, "return");
            auto t2 = high_resolution_clock::now();
            std::vector<torch::Tensor> outs = model_dict[agent_name]->forward(obss, true);
            auto t3 = high_resolution_clock::now();
            torch::Tensor probs1=outs[3];
            torch::Tensor probs2=outs[4];
            torch::Tensor values=outs[2];
            torch::Tensor idx1 =actions.select(1,0).to(torch::kLong);
            torch::Tensor idx2 =actions.select(1,1).to(torch::kLong);
            torch::Tensor prob1 = torch::diag(probs1.index_select(1, idx1));
            torch::Tensor prob2 = torch::diag(probs2.index_select(1, idx2));
            torch::Tensor prob = torch::stack({prob1, prob2},1);
            torch::Tensor entropy1 = -torch::sum(probs1*torch::log(probs1+EPSILON),-1);
            torch::Tensor entropy2 = -torch::sum(probs2*torch::log(probs2+EPSILON),-1);
            torch::Tensor entropy = torch::stack({entropy1, entropy2},1);
            
            torch::Tensor clipped_value= old_values + torch::clamp(values - old_values, -1 * epsilon, epsilon);
            torch::Tensor v_opt_a = (returns - values) * (returns - values);
            torch::Tensor v_opt_b = (returns - clipped_value) * (returns - clipped_value);
            torch::Tensor value_loss = 0.5 *torch::mean(torch::max(v_opt_a, v_opt_b));

            torch::Tensor theta = prob/(old_prob+EPSILON);
            torch::Tensor clipped_theta = torch::clamp(theta, 1.0 - epsilon, 1.0 + epsilon);
            torch::Tensor p_opt_a = theta * advantages;
            torch::Tensor p_opt_b = clipped_theta * advantages;
            torch::Tensor policy_loss = -1 * torch::mean(torch::min(p_opt_a, p_opt_b));
            torch::Tensor entropy_loss = -beta *torch::mean(entropy);
            torch::Tensor loss = policy_loss + value_loss + entropy_loss;
            avg_entropy_loss=avg_entropy_loss+entropy_loss.item().toFloat()/(float)total_update;
            avg_value_loss=avg_value_loss+value_loss.item().toFloat()/(float)total_update;
            avg_policy_loss=avg_policy_loss+policy_loss.item().toFloat()/(float)total_update;

            // std::cout<<"obss"<<std::endl<<obss<<std::endl;
            // std::cout<<"actions"<<std::endl<<actions<<std::endl;
            // std::cout<<"advantages"<<std::endl<<advantages<<std::endl;
            // std::cout<<"old_prob"<<std::endl<<old_prob<<std::endl;
            // std::cout<<"old_values"<<std::endl<<old_values<<std::endl;
            // std::cout<<"returns"<<std::endl<<returns<<std::endl;

            // std::cout<<"probs1"<<std::endl<<probs1<<std::endl;
            // std::cout<<"probs2"<<std::endl<<probs2<<std::endl;
            // std::cout<<"values"<<std::endl<<values<<std::endl;
            // std::cout<<"prob1"<<std::endl<<prob1<<std::endl;
            // std::cout<<"prob2"<<std::endl<<prob2<<std::endl;
            // std::cout<<"prob"<<std::endl<<prob<<std::endl;
            // std::cout<<"entropy1"<<std::endl<<entropy1<<std::endl;
            // std::cout<<"entropy2"<<std::endl<<entropy2<<std::endl;
            // std::cout<<"entropy"<<std::endl<<entropy<<std::endl;
            // std::cout<<"clipped_value"<<std::endl<<clipped_value<<std::endl;
            // std::cout<<"v_opt_a"<<std::endl<<v_opt_a<<std::endl;
            // std::cout<<"v_opt_b"<<std::endl<<v_opt_b<<std::endl;
            // std::cout<<"value_loss"<<std::endl<<value_loss<<std::endl;
            // std::cout<<"theta"<<std::endl<<theta<<std::endl;
            // std::cout<<"p_opt_a"<<std::endl<<p_opt_b<<std::endl;
            // std::cout<<"p_opt_b"<<std::endl<<p_opt_b<<std::endl;
            // std::cout<<"policy_loss"<<std::endl<<policy_loss<<std::endl;
            // std::cout<<"loss"<<std::endl<<policy_loss<<std::endl;
            auto t4 = high_resolution_clock::now();
            optim_ptr->zero_grad();
            loss.backward();
            auto t5 = high_resolution_clock::now();
            optim_ptr->step();
            auto d1 = duration_cast<microseconds>(t2 - t1);
            auto d2 = duration_cast<microseconds>(t3 - t2);
            auto d3 = duration_cast<microseconds>(t4 - t3);
            auto d4 = duration_cast<microseconds>(t5 - t4);
            f1=f1+d1.count()/1000;
            f2=f2+d2.count()/1000;
            f3=f3+d3.count()/1000;
            f4=f4+d4.count()/1000;
        }
    }

    // std::cout<<"1:"<<(int)(f1)<<" 2:"<<(int)(f2)<<" 3:"<<(int)(f3)<<" 4:"<<(int)(f4)<<std::endl;
    
    torch::save(*optim_ptr,get_model_full_path(agent_name, true));
    torch::save(model_dict[agent_name],get_model_full_path(agent_name, false));
    Dictionary out;
    out["value"]=avg_value_loss;
    out["policy"]=avg_policy_loss;
    out["entropy"]=avg_entropy_loss;
    return out;
}

void CTorch::set_opti_params(Dictionary setting){
    if (setting.has("rl")){
        learning_rate=setting["rl"];
    }
    if (setting.has("mini_batch")){
        mini_batch_size=setting["mini_batch"];
    }
    if (setting.has("epoch")){
        epoch=setting["epoch"];
    }
    if (setting.has("epsilon")){
        epsilon=setting["epsilon"];
    }
    if (setting.has("beta")){
        beta=setting["beta"];
    }
}

void CTorch::_bind_methods() {
    ClassDB::bind_method(D_METHOD("create_model","agent_name","hiddens"), &CTorch::create_model);
    ClassDB::bind_method(D_METHOD("load_model","agent_name","hiddens"), &CTorch::load_model);
    ClassDB::bind_method(D_METHOD("get_action", "obs","agent_name","b_train"), &CTorch::get_action);
    ClassDB::bind_method(D_METHOD("train","train_data", "hiddens"), &CTorch::train);
    ClassDB::bind_method(D_METHOD("test_variant","data"), &CTorch::test_variant);
    ClassDB::bind_method(D_METHOD("set_opti_params","setting"), &CTorch::set_opti_params);
}

CTorch::CTorch() {
}