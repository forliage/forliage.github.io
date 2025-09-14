document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '9999'; // Ensure it's on top

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    // Data structures for trail and particles
    const trail = [];
    const particles = [];
    const trailMaxAge = 40;
    const particleMaxAge = 50;
    const particleSpeed = 2;

    // Color palette inspired by the image
    const colors = ['#A044FF', '#FF44CC', '#FFD700', '#FFFFFF'];

    document.addEventListener('mousemove', (e) => {
        trail.push({ x: e.clientX, y: e.clientY, age: 0 });

        // Create a burst of particles
        for (let i = 0; i < 5; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * particleSpeed;
            particles.push({
                x: e.clientX,
                y: e.clientY,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                age: 0,
                size: Math.random() * 3 + 1,
                color: colors[Math.floor(Math.random() * colors.length)]
            });
        }
    });

    function updateAndDraw() {
        ctx.clearRect(0, 0, width, height);

        // --- Update and draw trail ---
        if (trail.length > 1) {
            for (let i = trail.length - 1; i > 0; i--) {
                const point = trail[i];
                const prevPoint = trail[i - 1];

                point.age++;
                
                const opacity = Math.max(0, 1 - (point.age / trailMaxAge));
                const lineWidth = Math.max(1, 10 * (1 - (point.age / trailMaxAge)));

                const gradient = ctx.createLinearGradient(prevPoint.x, prevPoint.y, point.x, point.y);
                const startColor = hexToRgba(colors[i % colors.length], opacity);
                const endColor = hexToRgba(colors[(i - 1) % colors.length], opacity);
                gradient.addColorStop(0, startColor);
                gradient.addColorStop(1, endColor);

                ctx.beginPath();
                ctx.moveTo(prevPoint.x, prevPoint.y);
                ctx.lineTo(point.x, point.y);
                ctx.strokeStyle = gradient;
                ctx.lineWidth = lineWidth;
                ctx.lineCap = 'round';
                ctx.stroke();
            }
        }
        
        // Remove old trail points
        while (trail.length > 0 && trail[0].age > trailMaxAge) {
            trail.shift();
        }

        // --- Update and draw particles ---
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.age++;
            p.x += p.vx;
            p.y += p.vy;

            if (p.age > particleMaxAge) {
                particles.splice(i, 1);
            } else {
                const opacity = Math.max(0, 1 - (p.age / particleMaxAge));
                ctx.beginPath();
                ctx.fillStyle = hexToRgba(p.color, opacity);
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        requestAnimationFrame(updateAndDraw);
    }

    // Helper function to convert hex color to rgba for opacity
    function hexToRgba(hex, alpha) {
        let r = 0, g = 0, b = 0;
        if (hex.length == 4) { // #RGB
            r = parseInt(hex[1] + hex[1], 16);
            g = parseInt(hex[2] + hex[2], 16);
            b = parseInt(hex[3] + hex[3], 16);
        } else if (hex.length == 7) { // #RRGGBB
            r = parseInt(hex.substring(1, 3), 16);
            g = parseInt(hex.substring(3, 5), 16);
            b = parseInt(hex.substring(5, 7), 16);
        }
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    updateAndDraw();
});
