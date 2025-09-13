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
    canvas.style.zIndex = '999';

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const points = [];
    const maxAge = 50; // Trail length in frames
    let hue = 0;

    document.addEventListener('mousemove', (e) => {
        points.push({ x: e.clientX, y: e.clientY, age: 0, hue: hue });
        hue = (hue + 3) % 360; // Cycle through hues for rainbow effect
    });

    function draw() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 1; i < points.length; i++) {
            const prevPoint = points[i - 1];
            const point = points[i];
            const opacity = Math.max(0, 1 - (point.age / maxAge));
            
            ctx.beginPath();
            ctx.moveTo(prevPoint.x, prevPoint.y);
            ctx.lineTo(point.x, point.y);

            const gradient = ctx.createLinearGradient(prevPoint.x, prevPoint.y, point.x, point.y);
            gradient.addColorStop(0, `hsla(${prevPoint.hue}, 100%, 70%, ${Math.max(0, 1 - (prevPoint.age / maxAge))})`);
            gradient.addColorStop(1, `hsla(${point.hue}, 100%, 70%, ${opacity})`);
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 4;
            ctx.lineCap = 'round';
            ctx.stroke();
            
            point.age++;
        }
        
        // Remove old points
        while (points.length > 0 && points[0].age > maxAge) {
            points.shift();
        }
    }

    function animate() {
        draw();
        requestAnimationFrame(animate);
    }

    animate();
});
