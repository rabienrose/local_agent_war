extends Spatial

export (Array, NodePath) var players_path
export (Array, NodePath) var targets_path
export (NodePath) var scene_path
export (NodePath) var camera_path

var camera
var players=[]
var targets=[]
var scene_node
var agent_step=0.2
var cur_agent_step_time=0
var scene_size
var step_count=0
var max_step=100


func _ready():
    randomize()
    camera=get_node(camera_path)
    for item in players_path:
        players.append(get_node(item))
    for item in targets_path:
        targets.append(get_node(item))
    scene_node=get_node(scene_path)
    scene_size = scene_node.get_scene_size()
    var first_player=false
    for p in players:
        if first_player==false:
            p.b_train=true
            first_player=true
        p.init(self)
    for t in targets:
        t.init(self)
    reset_game()
    # print("create model")

func set_main(b_val):
    if b_val==false:
        camera.queue_free()
        visible=false

func get_rand_posi():
    var rand_x = (randf()-0.5)*(scene_size-1)*2
    var rand_y = (randf()-0.5)*(scene_size-1)*2
    return Vector3(rand_x, 0,rand_y)

func end_game():
    for p in players:
        if p.b_train:
            p.on_collect_trajectory(true)
            p.on_episode_end()
    reset_game()

func reset_game():
    step_count=0
    for p in players:
        p.reset(translation + get_rand_posi())
    for t in targets:
        var rand_hue = randf()
        var c=Color.from_hsv(rand_hue,1,1)
        t.set_color(c)
        t.set_picked(false)
        t.translation=get_rand_posi() 
    for p in players:
        if p.b_train:
            p.on_episode_begin()

func _physics_process(delta):
    cur_agent_step_time=cur_agent_step_time+delta
    if cur_agent_step_time<agent_step:
        return
    cur_agent_step_time=0

    for p in players:
        if p.infer_task==null:
            p.infer_task=Global.torch.regist_infer_task(Global.agent_name)
        var obs = p.on_obs()
        var act_data = Global.torch.fetch_infer_action(p.infer_task)
        
        # var act_data1 = p.on_hueristic()
        # act_data[0]=act_data1[0]
        # act_data[1]=act_data1[1]
        if act_data!=null:
            if p.b_train:
                p.on_collect_trajectory(false)
            p.on_action(act_data)
        Global.torch.request_infer_action(obs, p.infer_task, p.b_train)
    step_count=step_count+1

    var has_remain_target=false
    for t in targets:
        if t.picked==false:
            has_remain_target=true
            break
    if has_remain_target==false:
        end_game()
        return
    if step_count>max_step:
        end_game()
        return
    for p in players:
        if p.dead:
            end_game()
            return
    
    
    
    

