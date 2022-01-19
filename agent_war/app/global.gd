extends Node

signal on_get_new_traj_data

var player_count=1
var target_count=2
var target_obs_size=3
var player_obs_size
var total_obs_size
var torch=null
var hiddens=128

var trajectorys=[] 
var mutex
var reward_record={}
var reward_count={}

var agent_name="chamo1"

func collect_trajectory(traj_data):
    mutex.lock()
    trajectorys.append(traj_data)
    emit_signal("on_get_new_traj_data")
    mutex.unlock()

func collect_reward_stats(reward_info):
    for key in reward_info:
        if not key in reward_record:
            reward_record[key]=0
            reward_count[key]=0
    for key in reward_record:
        if key in reward_info:
            reward_record[key]=reward_record[key]+reward_info[key]
        reward_count[key]=reward_count[key]+1

func get_trajectorys():
    var new_array=[]
    mutex.lock()
    new_array = trajectorys
    trajectorys=[]
    mutex.unlock()
    return new_array
    
func clear_trajectorys():
    # mutex.lock()
    trajectorys=[]
    # mutex.unlock()

func _ready():
    # Engine.time_scale=20
    # Engine.iterations_per_second=120
    mutex = Mutex.new()
    player_obs_size= 4+target_obs_size*target_count
    total_obs_size=player_obs_size*player_count
            

