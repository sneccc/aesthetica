import raylib as rl
from raylib.colors import *
import math
# Import custom math functions
from rlmath import *

# Set up the window
screenWidth = 1280
screenHeight = 720

# Enable anti-aliasing and allow window resizing
rl.SetConfigFlags(rl.FLAG_MSAA_4X_HINT| rl.FLAG_WINDOW_RESIZABLE)
rl.InitWindow(screenWidth, screenHeight, b"raylib [shaders]")

# Set up the camera
camera = rl.ffi.new('struct Camera3D *', [
    [2, 12, 6],  # Position
    [0, .5, 0],  # Target
    [0, 1, 0],   # Up vector
    45,          # FOV
    rl.CAMERA_PERSPECTIVE
])

# Get the current directory and set paths for resources
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
texture_path = os.path.join(current_dir, 'resources/test.png')
shader_path = os.path.join(current_dir, 'resources/shaders/wave.fs')

# Load texture and shader
texture = rl.LoadTexture(texture_path.encode('utf-8'))
shader = rl.LoadShader(b"", shader_path.encode('utf-8'))

# Get locations of shader uniforms
secondsLoc = rl.GetShaderLocation(shader, b"secondes")
freqXLoc = rl.GetShaderLocation(shader, b"freqX")
freqYLoc = rl.GetShaderLocation(shader, b"freqY")
ampXLoc = rl.GetShaderLocation(shader, b"ampX")
ampYLoc = rl.GetShaderLocation(shader, b"ampY")
speedXLoc = rl.GetShaderLocation(shader, b"speedX")
speedYLoc = rl.GetShaderLocation(shader, b"speedY")
mousePosLoc = rl.GetShaderLocation(shader, b"mousePos")

# Set initial values for shader parameters
freqX = rl.ffi.new("float *", 25.0)
freqY = rl.ffi.new("float *", 25.0)
ampX = rl.ffi.new("float *", 12.0)
ampY = rl.ffi.new("float *", 12.0)
speedX = rl.ffi.new("float *", 8.0)
speedY = rl.ffi.new("float *", 8.0)
mousePos = rl.ffi.new("struct Vector2 *", [0, 0])

# Set screen size in shader
screenSize = rl.ffi.new("struct Vector2 *",[ rl.GetScreenWidth(), rl.GetScreenHeight() ])
rl.SetShaderValue(shader, rl.GetShaderLocation(shader, b"size"), screenSize, rl.SHADER_UNIFORM_VEC2)

# Set initial shader values
rl.SetShaderValue(shader, freqXLoc, freqX, rl.SHADER_UNIFORM_FLOAT)
rl.SetShaderValue(shader, freqYLoc, freqY, rl.SHADER_UNIFORM_FLOAT)
rl.SetShaderValue(shader, ampXLoc, ampX, rl.SHADER_UNIFORM_FLOAT)
rl.SetShaderValue(shader, ampYLoc, ampY, rl.SHADER_UNIFORM_FLOAT)
rl.SetShaderValue(shader, speedXLoc, speedX, rl.SHADER_UNIFORM_FLOAT)
rl.SetShaderValue(shader, speedYLoc, speedY, rl.SHADER_UNIFORM_FLOAT)
rl.SetShaderValue(shader, mousePosLoc, mousePos, rl.SHADER_UNIFORM_VEC2)
# Initialize time
seconds = rl.ffi.new("float *", 0.0)

# Set target FPS
rl.SetTargetFPS(60)

# Main game loop
while not rl.WindowShouldClose():
    
    # Update time and pass it to shader
    seconds[0] += rl.GetFrameTime()
    rl.SetShaderValue(shader, secondsLoc, seconds, rl.SHADER_UNIFORM_FLOAT)

    # Update mouse position and pass it to shader
    getPos = rl.GetMousePosition()
    normalized_mouse_x = getPos.x / screenWidth
    normalized_mouse_y = getPos.y / screenHeight
    mousePos.x = normalized_mouse_x
    mousePos.y = normalized_mouse_y
    rl.SetShaderValue(shader, mousePosLoc, mousePos, rl.SHADER_UNIFORM_VEC2)

    # Start drawing
    rl.BeginDrawing()

    # Clear background
    rl.ClearBackground(RAYWHITE)
    
    # Apply shader
    rl.BeginShaderMode(shader)

    # Draw texture tiled across the screen
    for i in range(0, screenWidth, texture.width):
        for j in range(0, screenHeight, texture.height):
            rl.DrawTexture(texture, i, j, WHITE)
    
    # End shader mode
    rl.EndShaderMode()
    
    # Draw FPS counter
    rl.DrawFPS(10, 10)
    
    # End drawing
    rl.EndDrawing()

# Close the window when the game loop ends
rl.CloseWindow()   