extends RigidBody

class_name Agent

var body_node:MeshInstance
var game
var need_reset_posi=false
var reset_posi
var b_train=false
var trajectory=[]
var rewards={}
var cul_rewards={}
var dead=false
var infer_task=null
var last_act_data=null
var last_obs_data=null

func set_color(c):
    var mat=SpatialMaterial.new()
    mat.albedo_color=c
    body_node.set_surface_material(0,mat)

func _ready():
    pass

func on_get_target( _target):
    add_reward(1,"score")

func on_obs():
    var obs_data=[]
    var targets = game.targets
    obs_data.append(translation.x/game.scene_size)
    obs_data.append(translation.z/game.scene_size)
    obs_data.append(linear_velocity.x/game.scene_size)
    obs_data.append(linear_velocity.z/game.scene_size)
    for tar in targets:
        var dat = tar.get_obs_data(translation)
        obs_data += dat
    return obs_data
    
func init(_game):
    game=_game

func on_action(act_data):
    # print(act_data)
    add_reward(-0.01, "time")
    var hori_impulse=int(act_data[0])
    var verti_impulse=int(act_data[1])
    if hori_impulse==0:
        apply_central_impulse(Vector3(0,0,-1))
    if hori_impulse==2:
        apply_central_impulse(Vector3(0,0,1))
    if verti_impulse==0:
        apply_central_impulse(Vector3(-1,0,0))
    if verti_impulse==2:
        apply_central_impulse(Vector3(1,0,0))

func on_hueristic():
    var act_data=[1,1]
    if Input.is_key_pressed(KEY_W):
        act_data[0]=0
    if Input.is_key_pressed(KEY_S):
        act_data[0]=2
    if Input.is_key_pressed(KEY_A):
        act_data[1]=0
    if Input.is_key_pressed(KEY_D):
        act_data[1]=2
    return act_data

func reset(posi):
    reset_posi=posi
    need_reset_posi=true

func _integrate_forces(state):
    if need_reset_posi:
        state.transform.origin=reset_posi
        state.linear_velocity = Vector3(0,0,0)
        need_reset_posi = false

func add_reward(val, reward_name):
    if not reward_name in rewards:
        rewards[reward_name]=0
    rewards[reward_name]=rewards[reward_name]+val
    if not reward_name in cul_rewards:
        cul_rewards[reward_name]=0
    cul_rewards[reward_name]=cul_rewards[reward_name]+val   

func on_episode_begin():
    dead=false

func on_episode_end():
    Global.collect_trajectory(trajectory)
    Global.collect_reward_stats(cul_rewards)
    cul_rewards={}
    trajectory=[]
    last_act_data=null
    last_obs_data=null

func on_collect_trajectory(b_end):
    if last_act_data==null:
        last_act_data=infer_task["a"]
        last_obs_data=infer_task["o"]
        return
    var all_reward=0
    for key in rewards:
        all_reward=all_reward+rewards[key]
    
    rewards={}
    var step_info={}
    step_info["obs"]=infer_task["o"]
    if b_end==false:
        var act_data = infer_task["a"]
        step_info["act"]=[act_data[0],act_data[1]]
        step_info["value"]=act_data[2]
        var prob1=[act_data[3],act_data[4],act_data[5]]
        var prob2=[act_data[6],act_data[7],act_data[8]]
        step_info["prob"]=[prob1,prob2]
        step_info["reward"]=all_reward
    else:
        step_info["value"]=0
    # print(step_info)
    trajectory.append(step_info)
    last_act_data=infer_task["a"]
    last_obs_data=infer_task["o"]

func _physics_process(_delta):
    if translation.x>game.scene_size or translation.x<-game.scene_size or translation.z>game.scene_size or translation.z<-game.scene_size:
        if game.step_count>1:
            if dead==false:
                add_reward(-0.1,"out")
                dead=true
            
