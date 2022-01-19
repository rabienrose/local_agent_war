extends Node

var semaphore
var training_thread
var infer_threads=[]
var infer_semaphores=[]
var cur_traj_count=0
var train_data=[]
var batch_size=20480
var c_torch
var last_train_time=0

var infer_thread_num=2

var max_infer_task=10000
var infer_tasks=[]

func get_next_available_ind(i_ind):
    var temp_ind=i_ind
    while true:
        i_ind=i_ind+1
        if temp_ind==i_ind:
            return -1
        if i_ind>=max_infer_task:
            i_ind=0
        if infer_tasks[i_ind]["s"]==0:
            return i_ind

func _ready():
    c_torch=CTorch.new()
    c_torch.set_opti_params({"mini_batch":4096,"epoch":3,"rl":0.00001,"beta":0.001})
    # c_torch.create_model(Global.agent_name,Global.hiddens)
    c_torch.load_model(Global.agent_name,Global.hiddens)
    Global.torch=self
    semaphore = Semaphore.new()
    training_thread = Thread.new()
    training_thread.start(self, "training_thread")
    for i in range(infer_thread_num):
        infer_semaphores.append(Semaphore.new())
        infer_tasks.append([])
        var t_thread = Thread.new()
        t_thread.start(self, "infer_thread_func",i)
        infer_threads.append(t_thread)
    Global.connect("on_get_new_traj_data", self, "notify_new_traj")

func discount_rewards(r, gamma=0.99):
    var discounted_r = []
    discounted_r.resize(r.size())
    for i in range(0,r.size()):
        discounted_r[i]=0
    var running_add = 0
    for t in range(r.size()-1, -1, -1):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

func get_gae(traj_data, gamma=0.99, lambd=0.95):
    var delta_ts=[]
    for i in range(0,traj_data.size()-1):
        var delta_t = traj_data[i]["reward"] + gamma * traj_data[i+1]["value"] - traj_data[i]["value"]
        delta_ts.append(delta_t)
    var advantage = discount_rewards(delta_ts, gamma * lambd)
    return advantage

func training_thread():
    while true:
        semaphore.wait()
        var trajs=Global.get_trajectorys()
        var new_traj_count=trajs.size()
        for i in range(new_traj_count):
            var traj_data=trajs[i]
            var traj_length = traj_data.size()
            var advantages = get_gae(traj_data)
            if train_data.size() >=batch_size:
                break
            for j in range(traj_length-1):
                var step_info={}
                step_info["obs"]=traj_data[j]["obs"].duplicate(true)
                step_info["action"]=traj_data[j]["act"].duplicate(true)
                step_info["value"]=traj_data[j]["value"]
                step_info["prob"]=[]
                for k in range(traj_data[j]["act"].size()):
                    var act=traj_data[j]["act"][k]
                    var prob=traj_data[j]["prob"][k][int(act)]
                    step_info["prob"].append(prob)
                step_info["advantage"]=advantages[j]
                step_info["return"]=advantages[j]+traj_data[j]["value"]
                train_data.append(step_info)
        
        var step_num=train_data.size()
        if step_num>=batch_size:
            # get_tree().paused=true
            var t1 = OS.get_ticks_msec()
            # print(train_data)
            var train_stats = c_torch.train(Global.agent_name, train_data, Global.hiddens)
            var t2 = OS.get_ticks_msec()
            var training_time=t2-t1
            var t_time =t2 - last_train_time
            last_train_time=t2
            var total=0
            var avg_rewords={}
            for key in Global.reward_record:
                avg_rewords[key]=float(Global.reward_record[key])/float(Global.reward_count[key])
                total=total+avg_rewords[key]
            avg_rewords["total"]=total
            avg_rewords["train_rate"]=str(int(float(training_time)/float(t_time)*100))+"%"
            avg_rewords["T"]=t2-t1
            avg_rewords["step_num"]=step_num
            print(avg_rewords)
            Global.reward_record={}
            Global.reward_count={}
            # get_tree().paused=false
            
            train_data=[]

func regist_infer_task(agent_name):
    var min_thread_id=-1
    var min_task_num=-1
    for i in range(infer_tasks.size()):
        if infer_tasks[i].size()<min_task_num or min_task_num==-1:
            min_thread_id=i
            min_task_num=infer_tasks[i].size()
    if min_thread_id==-1:
        min_thread_id=0
    var new_task={"t":min_thread_id,"s":0,"o":[],"a":[],"n":agent_name,"b_t":true}
    infer_tasks[min_thread_id].append(new_task)
    return new_task

func fetch_infer_action(task):
    if task["s"]==0:
        return null
    else:
        while task["s"]==1:
            print("infer blocked")
            pass
        task["s"]=0
        return task["a"]

func request_infer_action(obs, task, b_train):
    task["o"]=obs
    task["s"]=1
    task["b_t"]=b_train
    infer_semaphores[task["t"]].post()

func infer_thread_func(thread_id):
    while true:
        infer_semaphores[thread_id].wait()
        for i in range(infer_tasks[thread_id].size()):
            if infer_tasks[thread_id][i]["s"]==1:
                var act_data = c_torch.get_action(infer_tasks[thread_id][i]["o"],infer_tasks[thread_id][i]["n"],infer_tasks[thread_id][i]["b_t"])
                infer_tasks[thread_id][i]["a"]=act_data
                infer_tasks[thread_id][i]["s"]=2

func notify_new_traj():
    semaphore.post()
