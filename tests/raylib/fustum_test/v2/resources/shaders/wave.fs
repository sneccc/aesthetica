#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;
uniform vec4 colDiffuse;

// Output fragment color
out vec4 finalColor;

uniform float secondes;

uniform vec2 size;

uniform float freqX;
uniform float freqY;
uniform float ampX;
uniform float ampY;
uniform float speedX;
uniform float speedY;

// New uniform for mouse position
uniform vec2 mousePos;

void main() {
    float pixelWidth = 1.0 / size.x;
    float pixelHeight = 1.0 / size.y;
    float aspect = pixelHeight / pixelWidth;
    float boxLeft = 0.0;
    float boxTop = 0.0;

    // Compute distance to mouse
    float distanceToMouse = distance(fragTexCoord, mousePos);
    
    // Define the radius within which waves are active
    float radius = 1;
    float influence = 1.0 - smoothstep(radius - 0.02, radius + 0.02, distanceToMouse);

    vec2 p = fragTexCoord;
    // Apply wave distortion influenced by the mouse position
    p.x += cos((fragTexCoord.y - boxTop) * freqX / (pixelWidth * 750.0) + (secondes * speedX)) * ampX * pixelWidth * influence;
    p.y += sin((fragTexCoord.x - boxLeft) * freqY * aspect / (pixelHeight * 750.0) + (secondes * speedY)) * ampY * pixelHeight * influence;

    // Modify color based on distance to mouse
    vec4 mouseEffect = vec4(1.0) - smoothstep(0.0, 0.1, distanceToMouse) * colDiffuse;

    finalColor = texture(texture0, p) * mouseEffect * fragColor;
}
