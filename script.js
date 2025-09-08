window.onload = function() {
    const canvas = document.getElementById('petalsCanvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const music = document.getElementById('bg-music');
    const musicBtn = document.getElementById('music-toggle');
    const darkModeToggle = document.getElementById('dark-mode-toggle');

    // Function to update highlight.js theme
    function updateHighlightTheme(isDarkMode) {
        const themeLink = document.getElementById('highlight-theme-link');
        if (themeLink) {
            const lightTheme = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/rose-pine-dawn.min.css';
            const darkTheme = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/rose-pine-moon.min.css';
            themeLink.href = isDarkMode ? darkTheme : lightTheme;
        }
    }

    // Function to apply the saved theme
    function applyTheme() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
            if (darkModeToggle) darkModeToggle.textContent = '☀️'; // Sun icon for light mode
        } else {
            document.body.classList.remove('dark-mode');
            if (darkModeToggle) darkModeToggle.textContent = '🌙'; // Moon icon for dark mode
        }
        updateHighlightTheme(isDarkMode);
    }

    // Toggle dark mode
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => {
            const isDarkMode = document.body.classList.toggle('dark-mode');
            localStorage.setItem('darkMode', isDarkMode);
            darkModeToggle.textContent = isDarkMode ? '☀️' : '🌙';
            updateHighlightTheme(isDarkMode);
        });
    }

    // Apply theme on initial load
    applyTheme();

    const savedTime = parseFloat(localStorage.getItem('music-current-time') || '0');
    const shouldPlay = localStorage.getItem('music-paused') !== 'true';

    function restoreMusic() {
        music.currentTime = savedTime;
        if (shouldPlay) {
            const playPromise = music.play();
            if (playPromise) {
                playPromise.then(() => {
                    musicBtn.classList.remove('paused');
                }).catch(() => {
                    musicBtn.classList.add('paused');
                });
            }
        } else {
            musicBtn.classList.add('paused');
        }
    }

    if (music.readyState >= 2) {
        restoreMusic();
    } else {
        music.addEventListener('canplay', restoreMusic);
    }

    musicBtn.addEventListener('click', () => {
        if (music.paused) {
            music.play();
            musicBtn.classList.remove('paused');
        } else {
            music.pause();
            musicBtn.classList.add('paused');
        }
        localStorage.setItem('music-paused', music.paused);
    });

    music.addEventListener('timeupdate', () => {
        localStorage.setItem('music-current-time', music.currentTime);
    });

    window.addEventListener('beforeunload', () => {
        localStorage.setItem('music-current-time', music.currentTime);
        localStorage.setItem('music-paused', music.paused);
    });

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            localStorage.setItem('music-current-time', music.currentTime);
            localStorage.setItem('music-paused', music.paused);
        }
    });
    
    recordVisit();

    const sidebarContainer = document.getElementById('sidebar-container');
    if (sidebarContainer) {
        loadSidebar().then(initSidebarFeatures);
    } else {
        initSidebarFeatures();
    }

    const taglineEl = document.getElementById('tagline');
    const phrases = ['这里有forliage的学习笔记', '分享科研和学习中的idea', '书写自己的精彩的生活', '分享绚丽的爱'];
    let phraseIndex = 0;
    let charIndex = 0;
    let deleting = false;

    function typeLoop() {
        if (!taglineEl) return;
        taglineEl.textContent = phrases[phraseIndex].substring(0, charIndex);
        if (!deleting && charIndex < phrases[phraseIndex].length) {
            charIndex++;
            setTimeout(typeLoop, 150);
        } else if (!deleting) {
            deleting = true;
            setTimeout(typeLoop, 1000);
        } else if (deleting && charIndex > 0) {
            charIndex--;
            setTimeout(typeLoop, 80);
        } else {
            deleting = false;
            phraseIndex = (phraseIndex + 1) % phrases.length;
            setTimeout(typeLoop, 500);
        }
    }
    typeLoop();
    
    let petals = [];
    let leaves = [];
    const numPetals = 70; 
    const numLeaves = 30; 
    const numClouds = 5; // Number of clouds
    const petalColors = ["#FFB6C1", "#FFC0CB", "#FF69B4", "#FF1493", "#DB7093"]; 
    const leafColors = ['#6B8E23', '#556B2F', '#8FBC8F', '#3CB371', '#2E8B57']; 
    const cloudColors = ['#FFC0CB', '#FFB6C1', '#FFDAB9', '#FFDAE9']; // Adjusted cloud colors
    const heartColors = ['#FF69B4', '#FF1493', '#C71585', '#DB7093']; // Specific vibrant heart colors
    
    let clouds = [];
    let hearts = [];

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    if (canvas) {
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas(); // Initial resize
    }

    class Petal {
        constructor() {
            this.reset();
            this.y = Math.random() * canvas.height; // Start at random y position
        }

        reset() {
            this.x = Math.random() * canvas.width;
            this.y = -Math.random() * 20 - 10; // Start above the screen
            this.size = Math.random() * 10 + 5; // Size between 5 and 15
            this.speedY = Math.random() * 1.2 + 0.8; // Slightly increased petal speed
            this.speedX = Math.random() * 0.6 - 0.3; // Adjusted petal sway
            this.opacity = Math.random() * 0.5 + 0.3; 
            this.color = petalColors[Math.floor(Math.random() * petalColors.length)];
            this.spin = Math.random() * Math.PI; 
            this.spinSpeed = (Math.random() - 0.5) * 0.02; 
        }

        update() {
            this.y += this.speedY;
            this.x += this.speedX;
            this.spin += this.spinSpeed;

            // Simple sway effect by slightly changing horizontal speed
            if (Math.random() > 0.95) {
                this.speedX = Math.random() * 0.5 - 0.25; // Adjusted to match reduced speed range
            }
            
            // Reset petal if it goes off screen
            if (this.y > canvas.height + this.size || this.x < -this.size || this.x > canvas.width + this.size) {
                this.reset();
            }
        }

        draw() {
            ctx.save();
            ctx.translate(this.x + this.size / 2, this.y + this.size / 2);
            ctx.rotate(this.spin);
            ctx.beginPath();
            ctx.moveTo(0, -this.size * 0.8);
            ctx.quadraticCurveTo(this.size * 0.6, -this.size * 0.6, 0, 0);
            ctx.quadraticCurveTo(-this.size * 0.6, -this.size * 0.6, 0, -this.size * 0.8);
            ctx.closePath();
            
            // Apply blur
            ctx.filter = 'blur(1px)'; 
            
            ctx.fillStyle = this.color;
            ctx.globalAlpha = this.opacity;
            ctx.fill();
            
            // Reset filter to avoid affecting other drawings or accumulating blur
            ctx.filter = 'none'; 
            
            ctx.restore();
        }
    }

    class Leaf {
        constructor() {
            this.reset();
            this.y = Math.random() * canvas.height; // Start at random y position
        }

        reset() {
            this.x = Math.random() * canvas.width;
            this.y = -Math.random() * 30 - 15; // Start slightly higher
            this.size = Math.random() * 12 + 8; // Leaves can be slightly larger
            this.speedY = Math.random() * 1.5 + 0.8; // Leaf speed
            this.speedX = Math.random() * 0.8 - 0.4; // Leaf sway
            this.opacity = Math.random() * 0.6 + 0.4; // Leaves can be a bit more opaque
            this.color = leafColors[Math.floor(Math.random() * leafColors.length)];
            this.spin = Math.random() * Math.PI;
            this.spinSpeed = (Math.random() - 0.5) * 0.015; // Slower spin for leaves
        }

        update() {
            this.y += this.speedY;
            this.x += this.speedX;
            this.spin += this.spinSpeed;

            // Wider sway for leaves
            if (Math.random() > 0.93) {
                this.speedX = Math.random() * 0.8 - 0.4; // Adjusted to match initial Leaf speedX range
            }
            
            if (this.y > canvas.height + this.size || this.x < -this.size || this.x > canvas.width + this.size) {
                this.reset();
            }
        }

        draw() {
            ctx.save();
            ctx.translate(this.x + this.size / 2, this.y + this.size / 2);
            ctx.rotate(this.spin);
            ctx.beginPath();
            ctx.moveTo(0, -this.size / 2);
            ctx.bezierCurveTo(this.size / 2, -this.size / 4, this.size / 2, this.size / 4, 0, this.size / 2);
            ctx.bezierCurveTo(-this.size / 2, this.size / 4, -this.size / 2, -this.size / 4, 0, -this.size / 2);
            ctx.closePath();
            
            ctx.filter = 'blur(1px)';
            ctx.fillStyle = this.color;
            ctx.globalAlpha = this.opacity;
            ctx.fill();
            ctx.filter = 'none';
            
            ctx.restore();
        }
    }

    function initBackgroundElements() {
        petals = []; 
        leaves = [];
        clouds = []; // Initialize clouds array
        hearts = []; // Clear hearts on resize too

        for (let i = 0; i < numPetals; i++) {
            petals.push(new Petal());
        }
        for (let i = 0; i < numLeaves; i++) {
            leaves.push(new Leaf());
        }
        for (let i = 0; i < numClouds; i++) {
            const sizeFactor = Math.random() * 0.7 + 0.8; // 0.8 to 1.5
            const cloudY = Math.random() * (canvas.height * 0.35) + (canvas.height * 0.05); 
            const speedX = (Math.random() * 0.3 + 0.15) * (Math.random() < 0.5 ? 1 : -1); 
            const initialX = (Math.random() * 1.5 - 0.25) * canvas.width; // Allows clouds to start off-screen
            clouds.push(new Cloud(
                initialX, 
                cloudY, 
                speedX, 
                sizeFactor, 
                cloudColors[Math.floor(Math.random() * cloudColors.length)]
            ));
        }
    }

    class Cloud {
        constructor(x, y, speedX, sizeFactor, color) {
            this.x = x;
            this.y = y;
            this.speedX = speedX;
            this.sizeFactor = sizeFactor;
            this.color = color;
            this.baseRadius = 25; // Increased base radius for fluffier clouds
            this.parts = [];
            // Define relative positions and sizes of circles that make up a cloud
            // More parts for a more complex cloud shape
            const partDefinitions = [
                { dx: 0, dy: 0, rScale: 1 },
                { dx: -25 * sizeFactor, dy: 5 * sizeFactor, rScale: 0.9 },
                { dx: 20 * sizeFactor, dy: 10 * sizeFactor, rScale: 0.95 },
                { dx: -15 * sizeFactor, dy: -15 * sizeFactor, rScale: 0.8 },
                { dx: 15 * sizeFactor, dy: -10 * sizeFactor, rScale: 0.85 },
                { dx: 40 * sizeFactor, dy: 0 * sizeFactor, rScale: 0.7 },
                { dx: -40 * sizeFactor, dy: -5 * sizeFactor, rScale: 0.65 }
            ];
            partDefinitions.forEach(pd => {
                this.parts.push({
                    dx: pd.dx,
                    dy: pd.dy,
                    r: this.baseRadius * pd.rScale * this.sizeFactor
                });
            });
            this.width = (this.baseRadius + 40 * sizeFactor) * 2; // Approximate width for wrapping logic
        }

        draw(ctx) {
            ctx.fillStyle = this.color;
            ctx.filter = 'blur(3px)'; // Slightly more blur for clouds
            this.parts.forEach(part => {
                ctx.beginPath();
                ctx.arc(this.x + part.dx, this.y + part.dy, part.r, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.filter = 'none';
        }

        update() {
            this.x += this.speedX;
            // If cloud is off-screen, reset its position
            if (this.speedX > 0 && this.x - this.width/2 > canvas.width) { // Moving right, off right edge
                this.x = -this.width/2;
                this.y = Math.random() * (canvas.height * 0.35) + (canvas.height * 0.05);
            } else if (this.speedX < 0 && this.x + this.width/2 < 0) { // Moving left, off left edge
                this.x = canvas.width + this.width/2;
                this.y = Math.random() * (canvas.height * 0.35) + (canvas.height * 0.05);
            }
        }
    }

    class Heart {
        constructor(x, y, size, color) {
            this.x = x;
            this.y = y;
            this.size = size;
            this.color = color;
            this.opacity = 1.0;
            this.life = 80 + Math.random() * 40; // Random life between 80-120 frames
            this.initialLife = this.life;
            this.isAlive = true;
            this.speedY = 0.3 + Math.random() * 0.3; // Slow downward drift for hearts
        }

        draw(ctx) {
            ctx.fillStyle = this.color;
            ctx.globalAlpha = this.opacity;
            ctx.filter = 'blur(1px)';

            const topCurveHeight = this.size * 0.3;
            ctx.beginPath();
            ctx.moveTo(this.x, this.y + topCurveHeight);
            ctx.bezierCurveTo(this.x, this.y, this.x - this.size / 2, this.y, this.x - this.size / 2, this.y + topCurveHeight);
            ctx.bezierCurveTo(this.x - this.size / 2, this.y + (this.size + topCurveHeight) / 1.8, this.x, this.y + this.size , this.x, this.y + this.size);
            ctx.bezierCurveTo(this.x, this.y + this.size , this.x + this.size / 2, this.y + (this.size + topCurveHeight) / 1.8, this.x + this.size / 2, this.y + topCurveHeight);
            ctx.bezierCurveTo(this.x + this.size / 2, this.y, this.x, this.y, this.x, this.y + topCurveHeight);
            ctx.closePath();
            ctx.fill();

            ctx.filter = 'none';
            ctx.globalAlpha = 1.0;
        }

        update() {
            this.life--;
            this.opacity = Math.max(0, (this.life / this.initialLife) * 0.8 + 0.2) ; // Fade but not to complete transparency too quickly
            this.y += this.speedY;
            this.size *= 0.99; // Slight shrink

            if (this.life <= 0 || this.opacity <= 0.05) { // Check opacity to ensure fade out
                this.isAlive = false;
            }
        }
    }


    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Clouds first (background)
        for (let cloud of clouds) {
            cloud.update();
            cloud.draw(ctx);
            // Spawn hearts from clouds
            if (Math.random() < 0.015) { // Adjusted probability
                const heartX = cloud.x + (Math.random() * cloud.width/3) - (cloud.width/6); 
                const heartY = cloud.y + cloud.baseRadius * cloud.sizeFactor * 0.3 + Math.random() * 10; 
                const heartSize = Math.random() * 8 + 12; 
                const heartColor = heartColors[Math.floor(Math.random() * heartColors.length)]; // Use specific heartColors
                hearts.push(new Heart(heartX, heartY, heartSize, heartColor));
            }
        }

        // Draw Petals and Leaves
        for (let petal of petals) {
            petal.update();
            petal.draw(ctx);
        }
        for (let leaf of leaves) {
            leaf.update();
            leaf.draw(ctx);
        }

        // Draw Hearts on top
        for (let i = hearts.length - 1; i >= 0; i--) {
            const heart = hearts[i];
            heart.update();
            heart.draw(ctx);
            if (!heart.isAlive) {
                hearts.splice(i, 1);
            }
        }

        requestAnimationFrame(animate);
    }

    if (canvas) {
        window.addEventListener('resize', () => {
            resizeCanvas();
            initBackgroundElements(); // Re-create petals and leaves on resize
        });

        initBackgroundElements();
        animate();
    }

    // 初始化侧栏功能（在 loadSidebar 完成后调用）
    function initSidebarFeatures() {
        updateArticleTitle();
        generateTOC(); // Add this call
        const progressContainer = document.querySelector('.progress-bar-container-placeholder');
        if (progressContainer) {
            progressContainer.className = 'progress-bar-container';
        }
        const progressBar = document.querySelector('.progress-bar-placeholder');
        if (progressBar) {
            progressBar.className = 'progress-bar';
        }
        window.addEventListener('scroll', updateReadingProgress);
        updateReadingProgress();
    }

    // Optional: Add new elements periodically if you want more than the initial set
    // setInterval(() => {
    //     if (petals.length < numPetals + 50) { // Add a cap
    //         petals.push(new Petal());
    //     }
    // }, 2000); // Add a new petal every 2 seconds

function updateArticleTitle() {
    const titlePlaceholder = document.getElementById('current-article-title-placeholder');
    let title = '';
    const mainArticle = document.querySelector('main article');
    if (mainArticle) {
        const heading = mainArticle.querySelector('h1, h2, h3, h4, h5, h6');
        if (heading) {
            title = heading.textContent;
        }
    }
    if (titlePlaceholder) {
        titlePlaceholder.textContent = title || '[无法获取标题]';
    }
}

function updateReadingProgress() {
    const progressBarElement = document.querySelector('.progress-bar');

    if (!progressBarElement) {
        console.log("Could not find progress bar element.");
        return;
    }

    const doc = document.documentElement;
    const scrollableHeight = doc.scrollHeight - doc.clientHeight;
    if (scrollableHeight <= 0) {
        progressBarElement.style.width = '0%';
        return;
    }

    const scrollTop = doc.scrollTop || document.body.scrollTop;
    const progressPercentage = (scrollTop / scrollableHeight) * 100;
    progressBarElement.style.width = progressPercentage + '%';
}

function generateTOC() {
    const tocContainer = document.getElementById('toc-container');
    const mainArticle = document.querySelector('main article');

    if (!tocContainer || !mainArticle) {
        return; // No container or article found, so no TOC needed.
    }

    const headings = mainArticle.querySelectorAll('h1, h2, h3, h4');
    
    if (headings.length === 0) {
        tocContainer.innerHTML = '<p style="font-size: 0.9em; color: #666;">本文无目录。</p>';
        return;
    }

    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';

    headings.forEach(heading => {
        if (!heading.id) {
            return; 
        }

        const listItem = document.createElement('li');
        const link = document.createElement('a');

        listItem.className = `toc-level-${heading.tagName.toLowerCase()}`;
        link.href = `#${heading.id}`;
        link.textContent = heading.textContent;

        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetElement = document.getElementById(heading.id);
            if(targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
    });

    tocContainer.appendChild(tocList);
}

    // Mouse click effect to show "鑫"
    document.addEventListener('mousedown', function showXinOnClick(event) {
        if (event.button !== 0) { // Only react to left mouse button
            return;
        }

        const xinElement = document.createElement('span');
        xinElement.textContent = '♥';

        // Adjust for centering (rough estimate, depends on font size)
        const fontSize = 24; // px
        const offsetX = fontSize / 2; 
        const offsetY = fontSize / 2; 

        xinElement.style.position = 'fixed';
        xinElement.style.left = (event.clientX - offsetX) + 'px';
        xinElement.style.top = (event.clientY - offsetY) + 'px';
        xinElement.style.color = 'red';
        xinElement.style.fontSize = fontSize + 'px';
        xinElement.style.pointerEvents = 'none';
        xinElement.style.userSelect = 'none';
        xinElement.style.transition = 'opacity 0.5s ease-out, transform 0.4s ease-out'; // Added transform for movement
        xinElement.style.opacity = '1';
        xinElement.style.transform = 'translateY(0)'; // Initial position for transform

        document.body.appendChild(xinElement);

        // Trigger fade out and upward movement
        setTimeout(() => {
            xinElement.style.opacity = '0';
            xinElement.style.transform = 'translateY(-20px)'; // Move upwards by 20px
        }, 50); // Start fade out and movement shortly after creation

        // Remove element after animation
        setTimeout(() => {
            if (xinElement.parentNode) {
                xinElement.parentNode.removeChild(xinElement);
            }
        }, 550); // Duration of opacity transition (500ms) + buffer
    });

    // Glass Effect Toggle
    const glassEffectToggle = document.getElementById('glass-effect-toggle');
    const blurSlider = document.getElementById('blur-slider');

    if (glassEffectToggle && blurSlider) {
        // Restore state from localStorage
        if (localStorage.getItem('glassEffectEnabled') === 'true') {
            document.body.classList.add('glass-effect-enabled');
            glassEffectToggle.textContent = 'Close';
            blurSlider.style.display = 'block';
        }

        const initialBlur = localStorage.getItem('blurIntensity') || '10';
        blurSlider.value = initialBlur;
        document.documentElement.style.setProperty('--blur-intensity', initialBlur + 'px');

        glassEffectToggle.addEventListener('click', () => {
            const isEnabled = document.body.classList.toggle('glass-effect-enabled');
            localStorage.setItem('glassEffectEnabled', isEnabled);

            if (isEnabled) {
                glassEffectToggle.textContent = 'Close';
                blurSlider.style.display = 'block';
            } else {
                glassEffectToggle.textContent = 'Blur';
                blurSlider.style.display = 'none';
            }
        });

        blurSlider.addEventListener('input', (e) => {
            const blurValue = e.target.value;
            document.documentElement.style.setProperty('--blur-intensity', blurValue + 'px');
            localStorage.setItem('blurIntensity', blurValue);
        });
    }
};

function recordVisit() {
    try {
        const list = JSON.parse(localStorage.getItem('visitHistory') || '[]');
        list.push({ time: new Date().toISOString(), page: location.pathname });
        localStorage.setItem('visitHistory', JSON.stringify(list));
    } catch (e) {
        console.error('Failed to record visit', e);
    }
}

function loadSidebar() {
    return new Promise((resolve) => {
        const container = document.getElementById('sidebar-container');
        if (!container) { resolve(); return; }
        let prefix = location.pathname.includes('_posts') ? '../' : './';
        fetch(prefix + 'sidebar.html')
            .then(r => r.text())
            .then(html => { container.innerHTML = html; resolve(); })
            .catch(err => { console.error('Failed to load sidebar', err); resolve(); });
    });
}

function sharePost(event, platform) {
    event.preventDefault();
    const postUrl = encodeURIComponent(window.location.href);
    const postTitle = encodeURIComponent(document.title);
    let shareUrl = '';

    switch (platform) {
        case 'weibo':
            shareUrl = `http://service.weibo.com/share/share.php?url=${postUrl}&title=${postTitle}`;
            break;
        case 'twitter':
            shareUrl = `https://twitter.com/intent/tweet?url=${postUrl}&text=${postTitle}`;
            break;
        case 'linkedin':
            shareUrl = `https://www.linkedin.com/shareArticle?mini=true&url=${postUrl}&title=${postTitle}`;
            break;
        case 'facebook':
            shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${postUrl}`;
            break;
        case 'qq':
            shareUrl = `http://connect.qq.com/widget/shareqq/index.html?url=${postUrl}&title=${postTitle}&summary=`;
            break;
        case 'wechat':
            alert('请在微信客户端打开或使用截图、二维码等方式进行分享。');
            return;
    }

    if (shareUrl) {
        window.open(shareUrl, '_blank', 'noopener,noreferrer,width=600,height=400');
    }
}