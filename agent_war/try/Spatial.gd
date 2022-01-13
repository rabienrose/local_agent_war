extends Spatial

func _ready():
    var torch = CTorch.new()
    torch.test_train()
