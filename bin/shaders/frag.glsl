#version 430
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D hdrTexture; // HDR texture for the skybox

uniform vec3 cameraPos;
uniform vec3 cameraDir;
uniform vec3 lightPos;
uniform vec2 screenSize; 

const float PI = 3.14159265;
const float rs = 1.0;

struct State
{
    vec4 pos; 
    vec4 vel; 
};

//TOOLS

float W_0(float x) //laaaambert
{
    if (x == 0) return 0;

    float e = exp(1);
    uint N = 3;

    float W = 1;

    if (x > e) {
        float l = log(x);
        W = l - log(l);
    } else if (x > 0) {
        W = x/e;
    } else {
        W = e*x*log(1+sqrt(1+e*x))/(2+e*x+sqrt(1+e*x));
    }

    for (uint n = 1; n <= N; n++) 
    {
        W = W*(1+log(x/W))/(1+W);
    }

    return W;
}

float ln(float x) { //why teh fuck odes this progrmaming lagnueage not have ln
    return log(x)/log(exp(1));
}

float atanh(float x) {
    return 0.5 * ln((1+x)/(1-x));
}

float DW_0(float x) { //derivative of lambert
    float W = W_0(x);
    return W/(x*(1+W));
}

//TRANSFORMS

vec4 SchwarzToCart(vec4 pos) {
    vec4 cart;
    cart.x = pos.x;
    cart.y = pos.y * sin(pos.z) * cos(pos.w);
    cart.z = pos.y * sin(pos.z) * sin(pos.w);
    cart.w = pos.y * cos(pos.z);
    
    return cart;
}

State SchwarzToCart(State state) {
    State cart = state;
    cart.pos.x = state.pos.x;
    cart.pos.y = state.pos.y * sin(state.pos.z) * cos(state.pos.w);
    cart.pos.z = state.pos.y * sin(state.pos.z) * sin(state.pos.w);
    cart.pos.w = state.pos.y * cos(state.pos.z);
    
    return cart;
}

vec4 CartToSchwarz(vec4 pos) {
    float yz = 1/(pos.y*pos.y+pos.z*pos.z);

    vec4 schwarz;
    schwarz.x = pos.x;
    schwarz.y = length(pos.yzw);
    schwarz.z = acos(pos.w/schwarz.y);
    schwarz.w = atan(pos.z,pos.y);

    return schwarz;
}

State CartToSchwarz(State state)
{

    State schwarz = state;
    schwarz.pos.x = state.pos.x;
    schwarz.pos.y = length(state.pos.yzw);
    schwarz.pos.z = acos(state.pos.w/schwarz.pos.y);
    schwarz.pos.w = atan(state.pos.z,state.pos.y);

    float x = state.pos.y;
    float y = state.pos.z;
    float z = state.pos.w;
    float r = schwarz.pos.y;

    mat3 J; //jacobian
    float xy = 1/(x*x+y*y);

    J[0][0] = x/r;
    J[0][1] = y/r;
    J[0][2] = z/r;

    J[1][0] = x*z*sqrt(xy)/(r*r);
    J[1][1] = y*z*sqrt(xy)/(r*r);
    J[1][2] = -sqrt(x*x+y*y)/(r*r);

    J[2][0] = -y*xy;
    J[2][1] = x*xy;
    J[2][2] = 0;

    schwarz.vel = vec4(state.vel.x, J*state.vel.yzw);

    return schwarz;
}

State SchwarzToKS(State state) 
{
    State KS = state;
    
    float r = state.pos.y;
    float t = state.pos.x;

    mat2 J;
    if (r>rs) {
        float factor = sqrt(r/rs-1)*exp(r/(2*rs));
        KS.pos.x = factor*sinh(t/(2*rs));
        KS.pos.y = factor*cosh(t/(2*rs));

        J[0][0] = KS.pos.y/(2*rs);
        J[0][1] = KS.pos.x*r/(2*rs*rs*(r/rs-1));
        J[1][0] = KS.pos.x/(2*rs);
        J[1][1] = KS.pos.y*r/(2*rs*rs*(r/rs-1));
    } else {
        float factor = sqrt(1-r/rs)*exp(r/(2*rs));
        KS.pos.x = factor*cosh(t/(2*rs));
        KS.pos.y = factor*sinh(t/(2*rs));

        J[0][0] = KS.pos.y/(2*rs);
        J[0][1] = -KS.pos.x*r/(2*rs*rs*(1-r/rs));
        J[1][0] = KS.pos.x/(2*rs);
        J[1][1] = -KS.pos.y*r/(2*rs*rs*(1-r/rs));
    }
    KS.pos.z = state.pos.z;
    KS.pos.w = state.pos.w;

    KS.vel.xy = J*state.vel.xy;
    KS.vel.z = state.vel.z;
    KS.vel.w = state.vel.w;

    return KS;
}

vec4 SchwarzToKS(vec4 pos) 
{
    vec4 KS;
    
    float r = pos.y;
    float t = pos.x;

    if (r>=rs) {
        float factor = sqrt(r/rs-1)*exp(r/(2*rs));
        KS.x = factor*sinh(t/(2*rs));
        KS.y = factor*cosh(t/(2*rs));
    } else {
        float factor = sqrt(1-r/rs)*exp(r/(2*rs));
        KS.x = factor*cosh(t/(2*rs));
        KS.y = factor*sinh(t/(2*rs));
    }
    KS.z = pos.z;
    KS.w = pos.w;

    return KS;
}

State KSToSchwarz(State state) 
{
    State Schwarz = state;

    float T = state.pos.x;
    float X = state.pos.y;
    
    Schwarz.pos.y = rs * (1+W_0((X*X-T*T)/exp(1)));
    if (T*T-X*X < 0) {
        Schwarz.pos.x = 2 * rs * atanh(T / X);
    } else {
        Schwarz.pos.x = 2 * rs * atanh(X / T);
    }
    Schwarz.pos.z = state.pos.z;
    Schwarz.pos.w = state.pos.w;

    Schwarz.vel = state.vel;

    mat2 J;
    J[0][0] = -2*X*rs/(T*T-X*X);
    J[0][1] = 2*T*rs/(T*T-X*X);
    J[1][0] = -2*T*rs*DW_0((X*X-T*T)/exp(1))/exp(1);
    J[1][1] = 2*X*rs*DW_0((X*X-T*T)/exp(1))/exp(1);

    Schwarz.vel.xy = J*state.vel.xy;

    return Schwarz;
}

vec4 KSToSchwarz(vec4 pos) 
{
    vec4 Schwarz;

    float T = pos.x;
    float X = pos.y;
    
    Schwarz.y = rs * (1+W_0((X*X-T*T)/exp(1)));
    if (T*T-X*X < 0) {
        Schwarz.x = 2 * rs * atanh(T / X);
    } else {
        Schwarz.x = 2 * rs * atanh(X / T);
    }
    Schwarz.z = pos.z;
    Schwarz.w = pos.w;

    return Schwarz;
}

//SDFs

float sphereSDF(vec3 p, vec3 c, float r) {
    return length(p - c) - r;
}

float sceneSDF(vec3 p) {
    return sphereSDF(p,vec3(0.0,0.0,0.0),1.0),sphereSDF(p,vec3(0.8,0.0,0.0),1.0);
}

//Metric comp

mat4 Metric(State s) {


    mat4 metric = mat4(1.0);
    metric[0][0] = -1.0;

    float r = s.pos.y;
    float theta = s.pos.z;
    float phi = s.pos.w;

   /* metric[0][0] = -(1-rs/r);
    metric[1][1] = 1/(1-rs/r);
    metric[2][2] = r * r;
    metric[3][3] = metric[2][2] * sin(theta) * sin(theta);*/

    return metric;
}

float partialg(State state, int mu, int nu, int direction)
{
    vec4 pos = state.pos;

    float delta = 0.01;
    
    State forward = state;
    forward.pos = pos + vec4(((direction == 0) ? delta : 0.0),
    ((direction == 1) ? delta : 0.0),
    ((direction == 2) ? delta : 0.0),
    ((direction == 3) ? delta : 0.0));
    
    State backward = state;
    backward.pos = pos - vec4(((direction == 0) ? delta : 0.0),
    ((direction == 1) ? delta : 0.0),
    ((direction == 2) ? delta : 0.0),
    ((direction == 3) ? delta : 0.0));

    mat4 gFor = Metric(forward);
    mat4 gBac = Metric(backward);

    return (gFor[mu][nu] - gBac[mu][nu]) / (2 * delta);
}

void Gamma(State state, out float gamma[4][4][4])
{
    mat4 g = Metric(state);
    mat4 ginv = inverse(g);

    for (int lambda = 0; lambda < 4; lambda++)
    {
        for (int mu = 0; mu < 4; mu++)
        {
            for (int nu = 0; nu < 4; nu++)
            {
                float gamma_value = 0.0;

                for (int sigma = 0; sigma < 4; sigma++)
                {
                    gamma_value += ginv[lambda][sigma] * (
                        partialg(state,nu, sigma, mu) +
                        partialg(state,mu, sigma, nu) -
                        partialg(state,nu, mu, sigma)
                    );
                }
                gamma[lambda][mu][nu] = 0.5 * gamma_value;
            }
        }
    }
}

State GeodesicEquation(State s)
{   
    
    float gamma[4][4][4];
    Gamma(s, gamma);
    
    s.pos = s.vel;
    
    vec4 accel = vec4(0.0, 0.0, 0.0, 0.0);
    for (int lambda = 0; lambda < 4; lambda++)
    {
        for (int mu = 0; mu < 4; mu++)
        {
            for (int nu = 0; nu < 4; nu++)
            {
                accel[lambda] -= gamma[lambda][mu][nu] * s.vel[mu] * s.vel[nu];
            }
        }
    }
    s.vel = accel;

    return s;
}

void Integrate(inout State s, float dt) {
    /*float3 org = { 0.0, 0.0, 8.0 };
    float3 force = 0.3 * normalize(state.pos - org) / pow(length(org - state.pos), 2);
    state.vel += dt * force;
    state.pos += dt * state.vel;*/ //Old
    
    State k1 = GeodesicEquation(s);
    /*State h1;
    h1.pos = s.pos + 0.5 * dt * k1.pos;
    h1.vel = s.vel + 0.5 * dt * k1.vel;
    State k2 = GeodesicEquation(h1);
    State h2;
    h2.pos = s.pos + 0.5 * dt * k2.pos;
    h2.vel = s.vel + 0.5 * dt * k2.vel;
    State k3 = GeodesicEquation(h2);
    State h3;
    h3.pos = s.pos + dt * k3.pos;
    h3.vel = s.vel + dt * k3.vel;
    State k4 = GeodesicEquation(h3);*/
    //to switch to rk4, uncomment previous comment and comment out the euler updating in exchange for rk4 updating

    s.vel += dt * k1.vel;
    s.pos += dt * k1.pos; //note that k1.pos is the velocity as k1 is a derivative
    //s.pos = s.pos + (dt / 6.0) * (k1.pos + 2.0 * k2.pos + 2.0 * k3.pos + k4.pos);
    //s.vel = s.vel + (dt / 6.0) * (k1.vel + 2.0 * k2.vel + 2.0 * k3.vel + k4.vel);
}

float raymarch(vec3 ro, vec3 rd, out vec3 hitPoint, out vec3 hitNormal) {
    float dt = 0.02;
    const int MAX_STEPS = 528;
    const float MAX_DIST = 100.0;
    const float EPSILON = 0.03;

    State cartS;
    cartS.pos = vec4(0.0,ro);
    cartS.vel = vec4(1.0,rd);

    State s = cartS;

    for (int i = 0; i < MAX_STEPS; i++) {
        //vec3 p = SchwarzToCart(s.pos).yzw;
        vec3 p = s.pos.yzw;
        float dist = sceneSDF(p); 

        if (dist < EPSILON) {
            hitPoint = p;
            vec2 e = vec2(0.01, 0.0);
            hitNormal = normalize(vec3(
                sceneSDF(p + e.xyy) - sceneSDF(p - e.xyy),
                sceneSDF(p + e.yxy) - sceneSDF(p - e.yxy),
                sceneSDF(p + e.yyx) - sceneSDF(p - e.yyx)
            ));
            return 1.0;
        }

        //s.pos += s.vel*dt;
        Integrate(s, dt);
        //t += dist;
        //if (t > MAX_DIST) break;
    }

    hitPoint = s.pos.yzw;
    return -1.0; // No hit
}

vec3 lighting(vec3 point, vec3 normal, vec3 lightPos) {
    vec3 lightDir = normalize(lightPos - point);
    float diffuse = max(dot(normal, lightDir), 0.0);
    return vec3(diffuse);
}

void main() {
    float aspectRatio = screenSize.x / screenSize.y;

    float fov = 150.0; 
    float scale = tan(radians(fov * 0.5));

    vec2 uv = (TexCoords - 0.5) * vec2(aspectRatio * scale, scale);
    vec3 ro = cameraPos;
    vec3 rd = normalize(cameraDir + vec3(uv, 1.0));

    vec3 hitPoint, hitNormal;
    float t = raymarch(ro, rd, hitPoint, hitNormal);

    if (t > 0.0) {
        vec3 color = lighting(hitPoint, hitNormal, lightPos);
        FragColor = vec4(color, 1.0);
    } else {
        vec3 dir = normalize(hitPoint-ro); 

        float theta = acos(dir.y); 
        float phi = atan(dir.z, dir.x);

        float u = (phi + PI) / (2.0 * PI); 
        float v = theta / PI;              

        FragColor = vec4(texture(hdrTexture, vec2(u, v)).rgb, 1.0); 
    }

}

