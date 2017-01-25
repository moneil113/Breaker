#include <cassert>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>

#include "Camera.h"
#include "GLSL.h"
#include "MatrixStack.h"
#include "ParticleSim.h"
#include "Program.h"
#include "Texture.h"
#include "Shape.h"

using namespace std;
using namespace Eigen;

GLFWwindow *window; // Main application window
string RESOURCE_DIR = "./"; // Where the resources are loaded from

shared_ptr<Camera> camera;
shared_ptr<Program> prog;
shared_ptr<Program> flatProg;
shared_ptr<Texture> texture0;
shared_ptr<ParticleSim> sim;
shared_ptr<Shape> box1;
shared_ptr<Shape> box2;
shared_ptr<Shape> box3;
Vector3f grav;
float t, dt;
float rotation; // rotation in degrees

bool keyToggles[256] = {false}; // only for English keyboards!

void rotateGravity(float delta) {
    Matrix3f rot;
    float deg = delta * 3.14159 / 180.0;
    rot << cos(deg), -sin(deg), 0,
    sin(deg), cos(deg), 0,
    0, 0, 1.0;
    grav = rot * grav;
    rotation += delta;
}

// This function is called when a GLFW error occurs
static void error_callback(int error, const char *description) {
	cerr << description << endl;
}

// This function is called when a key is pressed
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
    
    if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        rotateGravity(1);
    }
    if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        rotateGravity(-1);
    }
}

// This function is called when the mouse is clicked
static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if(action == GLFW_PRESS) {
		bool shift = (mods & GLFW_MOD_SHIFT) != 0;
		bool ctrl  = (mods & GLFW_MOD_CONTROL) != 0;
		bool alt   = (mods & GLFW_MOD_ALT) != 0;
		camera->mouseClicked((float)xmouse, (float)ymouse, shift, ctrl, alt);
	}
}

// This function is called when the mouse moves
static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse) {
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if(state == GLFW_PRESS) {
		camera->mouseMoved((float)xmouse, (float)ymouse);
	}
}

static void char_callback(GLFWwindow *window, unsigned int key) {
	keyToggles[key] = !keyToggles[key];
    if (key == 'r') {
        sim->reInit();
    }
    if (key == 's') {
        grav << 0.0f, -9.8f, 0.0f;
        rotation = 0;
    }
    if (key == '1') {
        keyToggles[(unsigned) '3'] = false;
        keyToggles[(unsigned) '1'] = true;
    }
    if (key == '3') {
        keyToggles[(unsigned) '1'] = false;
        keyToggles[(unsigned) '2'] = false;
        keyToggles[(unsigned) '3'] = true;
    }
}

// If the window is resized, capture the new size and reset the viewport
static void resize_callback(GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

// This function is called once to initialize the scene and OpenGL
static void init() {
	// Initialize time.
	glfwSetTime(0.0);
	
	// Set background color.
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	// Enable z-buffer test.
	glEnable(GL_DEPTH_TEST);
	// Enable setting gl_PointSize from vertex shader
	glEnable(GL_PROGRAM_POINT_SIZE);
    // Enable using gl_PointCoord
    glEnable(GL_POINT_SPRITE);

	prog = make_shared<Program>();
	prog->setShaderNames(RESOURCE_DIR + "vert.glsl", RESOURCE_DIR + "frag.glsl");
	prog->setVerbose(false);
	prog->init();
	prog->addAttribute("aPos");
	prog->addAttribute("aCol");
	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addUniform("screenSize");
	prog->addUniform("texture0");
	
	camera = make_shared<Camera>();
	camera->setInitDistance(9.0f);
	
	texture0 = make_shared<Texture>();
	texture0->setFilename(RESOURCE_DIR + "alpha.jpg");
	texture0->init();
	texture0->setUnit(0);
	texture0->setWrapModes(GL_REPEAT, GL_REPEAT);
	
	int n = 4096;
    sim = make_shared<ParticleSim>(n, 0.5);
	grav << 0.0f, -9.8f, 0.0f;
	t = 0.0f;
	dt = 0.01f;
    rotation = 0;
    
    flatProg = make_shared<Program>();
    flatProg->setShaderNames(RESOURCE_DIR + "flatVert.glsl", RESOURCE_DIR + "flatFrag.glsl");
    flatProg->setVerbose(false);
    flatProg->init();
    flatProg->addAttribute("aPos");
    flatProg->addUniform("P");
    flatProg->addUniform("MV");
    
    box1 = make_shared<Shape>();
    box1->loadMesh(RESOURCE_DIR + "box1.obj");
    box1->init();
    
    box2 = make_shared<Shape>();
    box2->loadMesh(RESOURCE_DIR + "box2.obj");
    box2->init();
    
    box3 = make_shared<Shape>();
    box3->loadMesh(RESOURCE_DIR + "box3.obj");
    box3->init();
    
    keyToggles[(unsigned) '1'] = true;
	
	GLSL::checkError(GET_FILE_LINE);
}

// This function is called every frame to draw the scene.
static void render() {
	// Clear framebuffer.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if(keyToggles[(unsigned)'l']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	camera->setAspect((float)width/(float)height);
	
	// Matrix stacks
	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	
	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
    MV->rotate(-rotation, Vector3f(0, 0, 1));
	camera->applyViewMatrix(MV);
	
    // Draw shapes
    if (keyToggles[(unsigned) '1'] || keyToggles[(unsigned) '3']) {
        MV->pushMatrix();
        MV->translate(Vector3f(3, 0.0f, 0));
        flatProg->bind();
        glUniformMatrix4fv(flatProg->getUniform("P"), 1, GL_FALSE, P->topMatrix().data());
        glUniformMatrix4fv(flatProg->getUniform("MV"), 1, GL_FALSE, MV->topMatrix().data());
        
        if (keyToggles[(unsigned) '1']) {
            box1->draw(flatProg);
            
            if (keyToggles[(unsigned) '2']) {
                box2->draw(flatProg);
            }
        }
        
        if (keyToggles[(unsigned) '3']) {
            box3->draw(flatProg);
        }
        
        // Draw the boundaries between particle buckets
        /*if (keyToggles[(unsigned) 'c']) {
            MV->pushMatrix();
            MV->translate(sim->mid() + Vector3f(3, 2, 0));
            glUniformMatrix4fv(flatProg->getUniform("MV"), 1, GL_FALSE, MV->topMatrix().data());
            cross->draw(flatProg);
            MV->popMatrix();
        }*/
        flatProg->unbind();
        MV->popMatrix();
    }
    
    // Draw particles
	glEnable(GL_BLEND);
	glDepthMask(GL_FALSE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	prog->bind();
	texture0->bind(prog->getUniform("texture0"));
	glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, P->topMatrix().data());
	glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, MV->topMatrix().data());
	glUniform2f(prog->getUniform("screenSize"), (float)width, (float)height);
    sim->draw(prog);
	texture0->unbind();
	prog->unbind();
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
	
	MV->popMatrix();
	P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}

void stepParticles() {
	if(keyToggles[(unsigned)' ']) {
        if (keyToggles[(unsigned) 's']) {
            rotateGravity(0.2);
        }
        sim->stepParticles(t, dt, grav, keyToggles);
		t += dt;
	}
}

int main(int argc, char **argv) {
	if(argc < 2) {
		cout << "Please specify the resource directory." << endl;
		return 0;
	}
	RESOURCE_DIR = argv[1] + string("/");

	// Set error callback.
	glfwSetErrorCallback(error_callback);
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(640, 480, "Breaker Fluid Simulation", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	// Initialize GLEW.
	glewExperimental = true;
	if(glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW" << endl;
		return -1;
	}
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	GLSL::checkVersion();
	// Set vsync.
	glfwSwapInterval(1);
	// Set keyboard callback.
	glfwSetKeyCallback(window, key_callback);
	// Set char callback.
	glfwSetCharCallback(window, char_callback);
	// Set cursor position callback.
	glfwSetCursorPosCallback(window, cursor_position_callback);
	// Set mouse button callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	// Set the window resize call back.
	glfwSetFramebufferSizeCallback(window, resize_callback);
	// Initialize scene.
	init();
    // Initialize frame rate counter
    double oldTime = glfwGetTime();
    double newTime;
	// Loop until the user closes the window.
	while(!glfwWindowShouldClose(window)) {
        newTime = glfwGetTime();
        cout << "fps: " << 1 / (newTime - oldTime) << endl;
        oldTime = newTime;
		// Step particles.
		stepParticles();
		// Render scene.
		render();
		// Swap front and back buffers.
		glfwSwapBuffers(window);
		// Poll for and process events.
		glfwPollEvents();
	}
	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
