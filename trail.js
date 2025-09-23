window.mouseTrailEffect = (function() {
    let canvas, ctx, style;
    let width, height;
    let points = [];
    let stars = [];
    const maxAge = 60;
    const starColors = ['#FFD6FF', '#FFF59D', '#FFFFFF'];
    const cursor = { x: -100, y: -100 };
    let animationFrameId;
    let isRunning = false;

    function setup() {
        if (canvas) return; // Already setup

        canvas = document.createElement('canvas');
        document.body.appendChild(canvas);
        ctx = canvas.getContext('2d');

        style = document.createElement('style');
        style.id = 'trail-cursor-style';
        style.textContent = 'body, a, button { cursor: none; }';
        document.head.appendChild(style);

        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '9999';

        window.addEventListener('resize', onResize);
        onResize(); // Set initial size

        document.addEventListener('mousemove', onMouseMove);
    }

    function onResize() {
        if (!canvas) return;
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    function onMouseMove(e) {
        cursor.x = e.clientX;
        cursor.y = e.clientY;

        if (!isRunning) return;
        
        points.push({ x: e.clientX, y: e.clientY, age: 0 });
        if (points.length > 60) points.shift();

        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 1 + 0.5;
        stars.push({
            x: e.clientX,
            y: e.clientY,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            age: 0,
            life: 80,
            size: Math.random() * 3 + 2,
            color: starColors[Math.floor(Math.random() * starColors.length)]
        });
    }

    function draw() {
        animationFrameId = requestAnimationFrame(draw);
        if (!ctx) return;
        ctx.clearRect(0, 0, width, height);

        if (isRunning) {
            for (const p of points) { p.age++; }
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.shadowBlur = 8;
            ctx.shadowColor = 'rgba(170,85,255,0.5)';

            for (let i = 1; i < points.length; i++) {
                const p = points[i];
                const prev = points[i - 1];
                const t = i / points.length;
                const r = Math.round(168 + (255 - 168) * t);
                const g = Math.round(75 + (182 - 75) * t);
                const b = 255;
                const alpha = t * 0.8;
                ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
                ctx.lineWidth = 8 * t;
                ctx.beginPath();
                ctx.moveTo(prev.x, prev.y);
                ctx.lineTo(p.x, p.y);
                ctx.stroke();
            }
            while (points.length && points[0].age > maxAge) {
                points.shift();
            }

            for (let i = stars.length - 1; i >= 0; i--) {
                const s = stars[i];
                s.age++;
                s.x += s.vx;
                s.y += s.vy;
                if (s.age > s.life) {
                    stars.splice(i, 1);
                    continue;
                }
                const opacity = 1 - s.age / s.life;
                drawStar(s.x, s.y, s.size, s.color, opacity);
            }
        }
        
        if (style && !style.disabled) {
            drawStar(cursor.x, cursor.y, 8, '#FFD700', 1);
        }
    }

    function drawStar(x, y, radius, color, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
            const outer = (18 + i * 72) * Math.PI / 180;
            const inner = (54 + i * 72) * Math.PI / 180;
            ctx.lineTo(x + Math.cos(outer) * radius, y + Math.sin(outer) * radius);
            ctx.lineTo(x + Math.cos(inner) * radius / 2, y + Math.sin(inner) * radius / 2);
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    function start() {
        if (isRunning) return;
        isRunning = true;
        if (style) style.disabled = false;
    }

    function stop() {
        if (!isRunning) return;
        isRunning = false;
        if (style) style.disabled = true;
        points = [];
        stars = [];
    }

    document.addEventListener('DOMContentLoaded', () => {
        setup();
        draw();
    });

    return {
        start,
        stop,
        get isRunning() { return isRunning; }
    };
})();