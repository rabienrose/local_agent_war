extends Spatial

export (NodePath) var body_path
var body_node:MeshInstance
var picked:bool
var game
var shape:CollisionShape

func set_color(c):
    var mat=SpatialMaterial.new()
    mat.albedo_color=c
    body_node.set_surface_material(0,mat)

func _ready():
    shape=get_node("CollisionShape")
    body_node=get_node(body_path)
    picked=false

func set_picked(b_val):
    picked=b_val
    if b_val:
        shape.disabled=true
        visible=false
    else:
        shape.disabled=false
        visible=true

func init(_game):
    game=_game

func get_obs_data(player_posi):
    var obs_data=[0,0,0]
    obs_data[0]=translation.x-player_posi.x
    obs_data[1]=translation.z-player_posi.z
    if picked:
        obs_data[2]=1
    return obs_data

func _on_Target_body_entered(body):
    set_picked(true)
    Global.emit_signal("on_get_target",body,self)
