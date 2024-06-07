#include "screenDisplay.h"

void window_size_callback(GLFWwindow* window, int width, int height)
{
    ScreenDisplay::updated = true;
    ScreenDisplay::m_screenSize = vec2i(width,height);
}
void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
        ScreenDisplay::translation = true;        
    }
    else if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE){
        ScreenDisplay::translation = false; 
    }
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
        ScreenDisplay::rotation = true;        
    }
    else if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
        ScreenDisplay::rotation = false; 
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if( !(ScreenDisplay::ihmpos.x < xpos && ScreenDisplay::ihmpos.x + ScreenDisplay::ihmsize.x > xpos && ScreenDisplay::ihmpos.y < ypos && ScreenDisplay::ihmpos.y + ScreenDisplay::ihmsize.y > ypos)){
        const vec2f t = ScreenDisplay::oldCursorPosition - vec2f(xpos,ypos);
        if( ScreenDisplay::rotation){
            ScreenDisplay::coordonneeSpherique = ScreenDisplay::coordonneeSpherique + t;
        }
        if(ScreenDisplay::translation){
            const vec3f atVector = normalize(ScreenDisplay::m_camera.at - ScreenDisplay::m_camera.pos);
            const vec3f rightVector = normalize(cross(atVector,ScreenDisplay::m_camera.up));
            const vec3f dt = vec3f(t.x * 0.005f,t.y * 0.005f, 0.f);
            ScreenDisplay::translateCamera = ScreenDisplay::translateCamera + rightVector*dt - ScreenDisplay::m_camera.up*dt;
        }
    }
    ScreenDisplay::oldCursorPosition = vec2f(xpos,ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    const vec3f atVector = normalize(ScreenDisplay::m_camera.at - ScreenDisplay::m_camera.pos);
    ScreenDisplay::m_camera.pos = ScreenDisplay::m_camera.pos + atVector*yoffset*0.5f;
}



ScreenDisplay::ScreenDisplay(const int width, const int height, const std::string title) : m_windowTitle(title){
    ScreenDisplay::updated = false;
    ScreenDisplay::m_screenSize = vec2i(width,height);

    if (!glfwInit())
    {
        // Initialization failed
        std::cerr << "GLFW initialization failed..." << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    
    window = glfwCreateWindow(m_screenSize.x, m_screenSize.y, m_windowTitle.c_str(), NULL, NULL);
    if (!window)
    {
        // Window or OpenGL context creation failed
        std::cerr << "GLFW window's creation failed..." << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    //Initialisation of Imgui
   // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.DisplaySize.x = width;io.DisplaySize.y = height;
    io.Fonts->Build();
   
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    const char* glsl_version = "#version 130";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


    pixels.resize(width*height);
    //Creation des objets de la scene
    createSceneEntities();  

    //Création du render
    optixRender = new OptixRender(&scene);
    optixRender->resize(m_screenSize);
}


ScreenDisplay::~ScreenDisplay(){
    delete(optixRender);
    glfwDestroyWindow(window);
    glfwTerminate();
}


void ScreenDisplay::createSceneEntities(){
    //Creation de la scene a afficher
    //Ajout d'un cube unitaire dans la scene
    MeshLoader loader;
    loader.LoadMesh("../data/statue/statue.obj");
    std::vector<TriangleMesh*> meshs;
    meshs = loader.toTriangleMeshs();
    for(size_t i = 0; i < meshs.size(); ++i)
        scene.addMesh(meshs[i]);
    Volume *v = new Volume();
    v->loadVolume("../data/cafard.dat");
    v->translate(vec3f(1.f,0.f,0.f));
    scene.addVolume(v);
}

void ScreenDisplay::run(){
    while (!glfwWindowShouldClose(window))
    {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        updateInterface();
        update();
        render();
        drawScene();
        
        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        //keep running
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void ScreenDisplay::updateInterface(){
    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    ImGui::Begin("Scene controler");
    
    if(ImGui::Button("Load volume")){ 
        std::string volume_path;
        std::cout << "Enter path to load volume" << std::endl;
        std::cin >> volume_path;
        Volume *v = new Volume(volume_path);
        scene.addVolume(v);
        optixRender->notifyAabbObjectChanges();
        optixRender->notifySbtChanges();
    }
    if(ImGui::Button("clean scene")){
        scene.clear();
        optixRender->notifyMeshChanges();
        optixRender->notifyAabbObjectChanges();
        optixRender->notifySbtChanges();
    }
    if(ImGui::Button("add mesh")){
        TriangleMesh *m = new TriangleMesh();
        m->addUnitCube();
        scene.addMesh(m);
        optixRender->notifyMeshChanges();
        optixRender->notifySbtChanges();
    }
    LaunchParams *parameters = optixRender->getLaunchParams();
    ImGui::SliderFloat("Number of samples", &parameters->frame.sampler, 1, 1000);
    ImGui::SliderFloat("Min intensity", &parameters->frame.minIntensity, 0.f, 1.f);
    ImGui::SliderFloat("Max Intensity", &parameters->frame.maxIntensity, 1.f, 0.f);

    // Partie de l'interface graphique pour choisir le type du Ray Tracing utilisé
    static int mip_option = 4;
    const char* mip_items[] = { "Mean", "MinIP", "LMIP", "DEMIP" , "MIP"};
    ImGui::Text("MIP Options:");
    ImGui::ListBox("##MIPOptions", &mip_option, mip_items, IM_ARRAYSIZE(mip_items), 5);
    parameters->frame.renderType =static_cast<unsigned char>(mip_option);

    //
    ImVec2 pos = ImGui::GetWindowPos();
    ImVec2 size = ImGui::GetWindowSize();
    ScreenDisplay::ihmpos = vec2f(pos.x,pos.y);
    ScreenDisplay::ihmsize = vec2f(size.x,size.y);
    ImGui::End();

    
    
}
void ScreenDisplay::update(){
    Camera cam;
    const vec2f CS2Rad = ScreenDisplay::coordonneeSpherique * -3.14f/180.f;
    vec3f atVector = normalize(ScreenDisplay::m_camera.at - ScreenDisplay::m_camera.pos);

    //Set de la position spherique
    const float r = norme(ScreenDisplay::m_camera.at - ScreenDisplay::m_camera.pos);

    cam.pos.x = r * sin(CS2Rad.x) * cos(CS2Rad.y);
    cam.pos.y = r * sin(CS2Rad.x) * sin(CS2Rad.y);
    cam.pos.z = r * cos(CS2Rad.x);

    atVector = normalize(ScreenDisplay::m_camera.at - ScreenDisplay::m_camera.pos);
    const vec3f rightVector = normalize(cross(atVector,ScreenDisplay::m_camera.up));

    //Rotation du vector up de t*0.005f
    Matrix3x3 rotation;
    Matrix3x3 rotationx = rotation.rotationX(CS2Rad.x); 
    Matrix3x3 rotationy = rotation.rotationZ(CS2Rad.y);
    
    rotation = rotationy;
    //rotation = rotation.rotationX(-ScreenDisplay::coordonneeSpherique.y / 3.14f * 180.f) * rotation.rotationY(-ScreenDisplay::coordonneeSpherique.x / 3.14f * 180.f);
    cam.up = normalize(rotation * vec3f(0.f,1.f,0.f));
        
    //translate cam
    cam.pos = cam.pos + ScreenDisplay::translateCamera;
    cam.at  = cam.at  + ScreenDisplay::translateCamera;

    optixRender->setCamera(cam);

    if(updated){
        resize(m_screenSize.x, m_screenSize.y);
        updated =false;
    }
}
void ScreenDisplay::render(){
    optixRender->render();
    optixRender->downloadPixels(pixels.data());
 }

void ScreenDisplay::drawScene(){
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.f,0.f,0.f,1.f);
    if (fbTexture == 0)
        glGenTextures(1, &fbTexture);


    glBindTexture(GL_TEXTURE_2D, fbTexture);

    // set basic parameters
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_screenSize.x, m_screenSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, m_screenSize.x, m_screenSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)m_screenSize.x, 0.f, (float)m_screenSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)m_screenSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)m_screenSize.x, (float)m_screenSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)m_screenSize.x, 0.f, 0.f);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void ScreenDisplay::resize(const int width, const int height){
    m_screenSize.x = width;
    m_screenSize.y = height;
    optixRender->resize(m_screenSize);
    pixels.resize(width * height);
}

vec2i ScreenDisplay::getSize() const {
    return m_screenSize;
}
int   ScreenDisplay::getWidth()  const {
    return m_screenSize.x;
}
int   ScreenDisplay::getHeight() const {
    return m_screenSize.y;
}

