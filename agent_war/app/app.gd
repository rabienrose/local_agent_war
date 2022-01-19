extends Spatial

export (Resource) var game_res
export (NodePath) var games_root_path
var games_root
var game_num=300

func _ready():
    games_root=get_node(games_root_path)

func _physics_process(_delta):
    var game_count=games_root.get_child_count()
    if game_count<game_num:
        # print("game count: ",game_count)
        var instance = game_res.instance()
        instance.translation.x=game_count*20;
        games_root.add_child(instance)
        if game_count>0:
            instance.set_main(false)

func _process(_delta):
    if _delta>0.0333:
        print(_delta)

        


