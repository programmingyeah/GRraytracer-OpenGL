#version 430
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D hdrTexture; // HDR texture for the skybox

uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 lightPos;
uniform vec2 screenSize; // New uniform for screen size

// Signed Distance Function for a sphere
float sphereSDF(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

// Raymarching function
float raymarch(vec3 ro, vec3 rd, out vec3 hitPoint, out vec3 hitNormal) {
    float t = 0.0;
    const int MAX_STEPS = 128;
    const float MAX_DIST = 100.0;
    const float EPSILON = 0.001;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + t * rd;
        float dist = sphereSDF(p, vec3(0.0, 0.0, 0.0), 1.0); // Sphere at origin

        if (dist < EPSILON) {
            hitPoint = p;
            vec2 e = vec2(0.001, 0.0);
            hitNormal = normalize(vec3(
                sphereSDF(p + e.xyy, vec3(0.0), 1.0) - sphereSDF(p - e.xyy, vec3(0.0), 1.0),
                sphereSDF(p + e.yxy, vec3(0.0), 1.0) - sphereSDF(p - e.yxy, vec3(0.0), 1.0),
                sphereSDF(p + e.yyx, vec3(0.0), 1.0) - sphereSDF(p - e.yyx, vec3(0.0), 1.0)
            ));
            return t;
        }

        t += dist;
        if (t > MAX_DIST) break;
    }

    return -1.0; // No hit
}

// Basic lighting
vec3 lighting(vec3 point, vec3 normal, vec3 lightPos) {
    vec3 lightDir = normalize(lightPos - point);
    float diffuse = max(dot(normal, lightDir), 0.0);
    return vec3(diffuse);
}

void main() {
    // Compute the aspect ratio
    float aspectRatio = screenSize.x / screenSize.y;

    // FOV setup
    float fov = 120.0; // Field of View in degrees
    float scale = tan(radians(fov * 0.5));

    // Adjust UV coordinates based on aspect ratio and FOV
    vec2 uv = (TexCoords - 0.5) * vec2(aspectRatio * scale, scale);
    vec3 ro = cameraPos;
    vec3 rd = normalize(cameraDir + vec3(uv, 1.0));

    // Raymarching
    vec3 hitPoint, hitNormal;
    float t = raymarch(ro, rd, hitPoint, hitNormal);

    if (t > 0.0) {
        vec3 color = lighting(hitPoint, hitNormal, lightPos);
        FragColor = vec4(color, 1.0);
    } else {
        // Map ray direction to UV coordinates for the HDR texture (for skybox sampling)
        float phi = atan(rd.z, rd.x);
        float theta = acos(rd.y);
        float u = (phi + 3.14159265) / (2.0 * 3.14159265); // Normalize to [0, 1]
        float v = theta / 3.14159265; // Normalize to [0, 1]

        FragColor = vec4(texture(hdrTexture, vec2(u, v)).rgb, 1.0); // Sample the HDR texture
    }
}

