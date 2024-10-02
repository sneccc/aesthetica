from raylib import *

screenWidth = 800
screenHeight = 450

InitWindow(screenWidth, screenHeight, b"raylib [textures] example - image loading")

texture = LoadTexture(b"C:\\Users\\Daniel\\CodingProjects\\aesthetica\\data\\normalized_3d_test\\painting\\painting_1120.png")

while not WindowShouldClose():

    BeginDrawing()

    ClearBackground(RAYWHITE)
    
    #DrawTexture(texture, int(screenWidth/2 - texture.width/2), int(screenHeight/2 - texture.height/2), WHITE)
    DrawTextureEx(texture, (0, 0), 0, 0.1, WHITE)

    DrawText(b"this IS a texture loaded from an image!", 300, 370, 10, GRAY)

    EndDrawing()

UnloadTexture(texture)

CloseWindow()  