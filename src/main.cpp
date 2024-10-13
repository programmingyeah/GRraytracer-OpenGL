#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Camera properties
float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = -8.0f; // Initial camera position
float cameraSpeed = 0.25f/5; // Camera movement speed

void processInput(GLFWwindow *window) {
    // Handle WASD for forward/backward/left/right movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) // Move forward
        cameraPosZ += cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) // Move backward
        cameraPosZ -= cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) // Move left
        cameraPosX -= cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) // Move right
        cameraPosX += cameraSpeed;

    // Handle EQ for up/down movement
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) // Move up
        cameraPosY += cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) // Move down
        cameraPosY -= cameraSpeed;

    // Handle escape key to unlock the cursor
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // Show cursor
    }
}

// Function to load shader from file
std::string loadShaderSource(const char* filePath) {
    std::ifstream shaderFile(filePath);
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    return shaderStream.str();
}

GLuint loadHDRTexture(const char* filePath) {
    // Variables for texture dimensions and format
    int width, height, channels;

    // Load the HDR texture using stb_image
    float* data = stbi_loadf(filePath, &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load HDR texture: " << stbi_failure_reason() << std::endl;
        return 0; // Return 0 if loading failed
    }

    // Generate and bind the OpenGL texture
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload the texture data to OpenGL
    GLenum format = (channels == 3) ? GL_RGB : GL_RGBA; // Determine format
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_FLOAT, data);

    // Free the image data loaded by stb_image
    stbi_image_free(data);

    return textureID; // Return the generated texture ID
}

// Function to compile and link shaders
GLuint compileShaderProgram(const char* vertexPath, const char* fragmentPath) {
    // Load the vertex/fragment shader source from file
    std::string vertexCode = loadShaderSource(vertexPath);
    std::string fragmentCode = loadShaderSource(fragmentPath);
    const char* vertexShaderSource = vertexCode.c_str();
    const char* fragmentShaderSource = fragmentCode.c_str();

    // Compile the vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex Shader Compilation Failed:\n" << infoLog << std::endl;
    }

    // Compile the fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment Shader Compilation Failed:\n" << infoLog << std::endl;
    }

    // Link shaders into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader Program Linking Failed:\n" << infoLog << std::endl;
    }

    // Cleanup shaders (no longer needed after linking)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Callback function to handle window resizing
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Adjust the OpenGL viewport to the new window dimensions
    glViewport(0, 0, width, height);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW for OpenGL 3.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a GLFW window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Raymarching", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Load OpenGL functions using GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Compile shaders from external files
    GLuint shaderProgram = compileShaderProgram("shaders/vert.glsl", "shaders/frag.glsl");

    // Set the viewport to match the initial window size
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    // Register the framebuffer size callback to handle window resizing
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Define vertices for a full-screen quad
    float quadVertices[] = {
        -1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
    };

    // Set up vertex array and vertex buffer objects
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Set up uniform locations
    glUseProgram(shaderProgram);
    GLint cameraPosLoc = glGetUniformLocation(shaderProgram, "cameraPos");
    GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLint cameraDirLoc = glGetUniformLocation(shaderProgram, "cameraDir");
    GLint screenSizeLoc = glGetUniformLocation(shaderProgram, "screenSize");
    GLuint hdrTexture = loadHDRTexture("resources/rogland_clear_night_4k.hdr");

    if (hdrTexture == 0) {
        std::cerr << "Failed to load HDR texture!" << std::endl;
    }

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        // Get updated window size and pass it to the shader
        glfwGetFramebufferSize(window, &width, &height);
        glUniform2f(screenSizeLoc, (float)width, (float)height);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        glUniform3f(cameraPosLoc, cameraPosX, cameraPosY, cameraPosZ);
        glUniform3f(cameraDirLoc, 0, 0, 0.0); //note, camera directions literally just dont work
        glUniform3f(lightPosLoc, 2.0f, 2.0f, -2.0f);  

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "hdrTexture"), 0);

        // Use the shader program and draw the quad
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
