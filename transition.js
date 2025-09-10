// --- 1. Basic Setup ---
// Prevent flash of unstyled content
const preventFlash = () => {
    document.addEventListener('DOMContentLoaded', () => {
        // Add a class to the body to signify that JS is enabled and styles can be applied
        document.body.classList.add('js-enabled');
    });
};
preventFlash();


// --- 2. WebGL Scene Setup ---
let scene, camera, renderer, plane, material;

const initWebGL = () => {
    scene = new THREE.Scene();
    
    // Camera setup
    const fov = (180 * (2 * Math.atan(window.innerHeight / 2 / 800))) / Math.PI; // Adjust fov to keep plane filling screen
    camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.z = 800;

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);
    renderer.domElement.style.position = 'fixed';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.zIndex = '9999';
    renderer.domElement.style.pointerEvents = 'none'; // Make sure it doesn't block clicks
    renderer.domElement.style.display = 'none'; // Initially hidden

    // Geometry and Material
    const geometry = new THREE.PlaneGeometry(window.innerWidth, window.innerHeight, 1, 1);
    
    material = new THREE.ShaderMaterial({
        uniforms: {
            u_time: { value: 0.0 },
            u_progress: { value: 0.0 },
            u_mouse: { value: new THREE.Vector2(0.5, 0.5) },
            u_noise: { value: new THREE.TextureLoader().load('https://i.ibb.co/P9pB7J2/noise.png') } // A simple noise texture
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform float u_progress;
            uniform vec2 u_mouse;
            uniform sampler2D u_noise;
            
            varying vec2 vUv;

            void main() {
                float progress = u_progress;
                
                // Calculate distance from mouse position
                float dist = distance(vUv, u_mouse);

                // Create a circular wave that expands
                float wave = smoothstep(progress - 0.1, progress, dist) - smoothstep(progress, progress + 0.1, dist);

                // Get noise value
                vec4 noise = texture2D(u_noise, vUv);

                // Combine wave with noise for a splashy effect
                float finalEffect = wave * noise.r;

                // Make the effect transparent
                gl_FragColor = vec4(0.1, 0.1, 0.1, finalEffect); 
            }
        `,
        transparent: true
    });

    plane = new THREE.Mesh(geometry, material);
    scene.add(plane);
};


// --- 3. Animation & Barba.js Setup ---
const animate = () => {
    requestAnimationFrame(animate);
    material.uniforms.u_time.value += 0.01;
    renderer.render(scene, camera);
};

const setupBarba = () => {
    let mousePos = { x: 0.5, y: 0.5 };
    
    // Store mouse position for the shader
    window.addEventListener('mousemove', (e) => {
        mousePos.x = e.clientX / window.innerWidth;
        mousePos.y = 1.0 - (e.clientY / window.innerHeight);
    });

    barba.init({
        sync: true, // Ensure leave and enter animations overlap
        transitions: [{
            name: 'splash-transition',
            leave(data) {
                // Get the click position, default to center if not a click event
                let clickPos = data.trigger.clientX ? {x: data.trigger.clientX / window.innerWidth, y: 1.0 - (data.trigger.clientY / window.innerHeight)} : {x: 0.5, y: 0.5};
                material.uniforms.u_mouse.value.set(clickPos.x, clickPos.y);

                return new Promise(resolve => {
                    renderer.domElement.style.display = 'block';
                    gsap.to(material.uniforms.u_progress, {
                        duration: 1.5, // Animation duration
                        value: 1.0,
                        ease: 'power3.out',
                        onComplete: () => {
                            resolve();
                        }
                    });
                });
            },
            enter(data) {
                // Reset progress and hide the canvas
                gsap.fromTo(material.uniforms.u_progress, 
                    { value: 1.0 },
                    {
                        duration: 1.5,
                        value: 0.0,
                        ease: 'power3.inOut',
                        onComplete: () => {
                            renderer.domElement.style.display = 'none';
                        }
                    }
                );
            }
        }]
    });
};


// --- 4. Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Only init if WebGL is supported
    if (typeof THREE !== 'undefined') {
        initWebGL();
        animate();
        if (typeof barba !== 'undefined' && typeof gsap !== 'undefined') {
            setupBarba();
        } else {
            console.error("Barba.js or GSAP is not loaded.");
        }
    } else {
        console.error("Three.js is not loaded.");
    }
});
