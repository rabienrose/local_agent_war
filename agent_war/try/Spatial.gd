extends Spatial

var test_thread1
var test_thread2
var test_thread3
var test_train_thread
var obs=[-0.01936, -0.711444, -0.166241, 0.166241, -0.550327, 1.359689, 0, -0.467354, 1.324418, 0]
var train_data_single={"action":[2, 2], "advantage":-0.13122, "obs":[0.756354, -0.252317, 0, 0, -1.260451, -0.050595, 0, -0.862713, 0.27064, 0], "prob":[0.340503, 0.311286], "return":0.015023, "value":0.146242}
var train_data=[]
var throughput=0
var throughput_train=0
var time_cul=0
var torch
func _ready():
    torch=CTorch.new()
    torch.set_opti_params({"mini_batch":2048,"epoch":10})
    torch.create_model("chamo",Global.hiddens)
    torch.load_model(Global.agent_name,Global.hiddens)
    for i in range(2048):
        train_data.append(train_data_single)
    var torch = CTorch.new()
    test_thread1 = Thread.new()
    test_thread1.start(self, "test_thread")
    test_thread2 = Thread.new()
    test_thread2.start(self, "test_thread")
    # test_thread3 = Thread.new()
    # test_thread3.start(self, "test_thread")
    test_train_thread = Thread.new()
    test_train_thread.start(self, "test_train")
    
func _process(_d):
    time_cul=time_cul+_d
    if time_cul>=1:
        print(throughput/time_cul," ",throughput_train/time_cul)
        throughput=0
        throughput_train=0
        time_cul=0

func test_train():
    while true:
        Global.torch.train(Global.agent_name, train_data, Global.hiddens)
        throughput_train=throughput_train+1

func test_thread():
    while true:
        torch.get_action(obs,"chamo",true)
        throughput=throughput+1
