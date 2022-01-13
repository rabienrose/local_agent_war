extends Spatial

export (NodePath) var ground_path
var ground

func _ready():
    ground=get_node(ground_path)

func get_scene_size():
    return ground.mesh.size.x/2
