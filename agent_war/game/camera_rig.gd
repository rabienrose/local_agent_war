extends Spatial


# Control variables
export var maxPitch : float = 45
export var minPitch : float = -45
export var maxZoom : float = 20
export var minZoom : float = 4
export var zoomStep : float = 2
export var rot_sens : float = 0.002
export var camLerpSpeed : float = 16.0

# Private variables
var _cam
var _curZoom : float = 0.0
var pressed=false

func _ready():
    _cam = get_node("ClippedCamera")
    _cam.translate(Vector3(0,0,maxZoom))
    _curZoom = maxZoom

func _input(event):
    if event is InputEventMouseMotion:
        if pressed:
            rotate_y(-event.relative.x * rot_sens)
            rotation.x = clamp(rotation.x - event.relative.y * rot_sens, deg2rad(minPitch), deg2rad(maxPitch))
            orthonormalize()
        
    if event is InputEventMouseButton:
        # Change zoom level on mouse wheel rotation
        if event.is_pressed():
            if event.button_index ==BUTTON_LEFT:
                pressed=true
            if event.button_index == BUTTON_WHEEL_UP and _curZoom > minZoom:
                _curZoom -= zoomStep
            if event.button_index == BUTTON_WHEEL_DOWN and _curZoom < maxZoom:
                _curZoom += zoomStep
        else:
            if event.button_index ==BUTTON_LEFT:
                pressed=false

func _process(delta) -> void:
    _cam.set_translation(_cam.translation.linear_interpolate(Vector3(0,0,_curZoom),delta * camLerpSpeed))
