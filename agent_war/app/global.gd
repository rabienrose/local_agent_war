extends Node

signal on_get_target(player, target)

var player_count=1
var target_count=2
var target_obs_size=3
var player_obs_size
var total_obs_size

func _ready():
    player_obs_size= 2+target_obs_size*target_count
    total_obs_size=player_obs_size*player_count
            

