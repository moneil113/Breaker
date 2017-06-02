#version 120

// varying vec4 vCol;

void main() {
    vec4 vCol = vec4(0.5, 0.5, 0.8, 1.0);
    // adapted from NVIDIA sample to draw point sprites as spheres
    const vec3 lightDir = vec3(0.5, -0.1, 0.577);
    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord * 2.0 - vec2(1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    // discard pixels outside radius
    if (mag > 1.0) {
        discard;
    }

    N.z = sqrt(1.0 - mag);

    // calculate lighting
    vec4 ambient = vec4(vCol.xyz * 0.3, 1.0);;
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = vCol * diffuse + ambient;
}
