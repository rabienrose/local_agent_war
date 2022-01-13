extends RigidBody

class_name Agent

var body_node:MeshInstance
var game
var need_reset_posi=false
var reset_posi

func set_color(c):
    var mat=SpatialMaterial.new()
    mat.albedo_color=c
    body_node.set_surface_material(0,mat)

func _ready():
    pass

func on_obs():
    var obs_data=[]
    var targets = game.targets
    obs_data.append(translation.x)
    obs_data.append(translation.z)
    for tar in targets:
        var dat = tar.get_obs_data(translation)
        obs_data += dat
    return obs_data
    
func init(_game):
    game=_game

func on_action(act_data):
    var hori_impulse=act_data[0]
    var verti_impulse=act_data[1]
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

func _physics_process(delta):
    pass
