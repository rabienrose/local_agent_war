extends Spatial

export (Array, NodePath) var players_path
export (Array, NodePath) var targets_path
export (NodePath) var scene_path

var players=[]
var targets=[]
var training_thread
var semaphore
var scene_node
var agent_step=0.2
var cur_agent_step_time=0

func _ready():
    randomize()
    semaphore = Semaphore.new()
    training_thread = Thread.new()
    training_thread.start(self, "training_thread")
    for item in players_path:
        players.append(get_node(item))
    for item in targets_path:
        targets.append(get_node(item))
    scene_node=get_node(scene_path)
    Global.connect("on_get_target",self,"on_picked_target")
    for p in players:
        p.init(self)
    for t in targets:
        t.init(self)
    reset_game()
    var torch=CTorch.new()
    torch.create_model()

func training_thread():
    while true:
        semaphore.wait()

func get_rand_posi():
    var scene_size = scene_node.get_scene_size()-1
    var rand_x = (randf()-0.5)*scene_size*2
    var rand_y = (randf()-0.5)*scene_size*2
    return Vector3(rand_x, 0,rand_y)

func reset_game():
    for p in players:
        p.reset(get_rand_posi())
    for t in targets:
        var rand_hue = randf()
        var c=Color.from_hsv(rand_hue,1,1)
        t.set_color(c)
        t.set_picked(false)
        t.translation=get_rand_posi()

func on_picked_target(_player, _target):
    var has_remain_target=false
    for t in targets:
        if t.picked==false:
            has_remain_target=true
            break
    if has_remain_target==false:
        reset_game()

func _physics_process(delta):
    cur_agent_step_time=cur_agent_step_time+delta
    if cur_agent_step_time<agent_step:
        return
    cur_agent_step_time=0
    for p in players:
        # print(p.on_obs())
        var act_data = p.on_hueristic()
        p.on_action(act_data)
    
    

