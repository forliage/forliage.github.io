document.addEventListener('DOMContentLoaded', () => {
    const bar = document.createElement('div');
    bar.id = 'charging-bar';
    document.body.appendChild(bar);
    window.addEventListener('load', () => bar.remove());
});

window.onload = function() {
    const music = document.getElementById('bg-music');
    let musicBtn = document.getElementById('music-toggle');
    let darkModeToggle = document.getElementById('dark-mode-toggle');

    if (darkModeToggle && darkModeToggle.tagName !== 'INPUT') {
        const wrapper = document.createElement('label');
        wrapper.className = 'dark-mode-control liquid-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = 'dark-mode-toggle';
        const knob = document.createElement('span');
        knob.className = 'knob';
        wrapper.appendChild(checkbox);
        wrapper.appendChild(knob);
        darkModeToggle.replaceWith(wrapper);
        darkModeToggle = checkbox;
    } else if (darkModeToggle) {
        const parent = darkModeToggle.closest('label');
        if (parent && !parent.querySelector('.knob')) {
            const knob = document.createElement('span');
            knob.className = 'knob';
            darkModeToggle.after(knob);
        }
    }
    const darkToggleLabel = darkModeToggle ? darkModeToggle.closest('.dark-mode-control') : null;

    // Apple Music 风格迷你播放器
    const miniPlayer = document.createElement('div');
    miniPlayer.id = 'mini-player';
    miniPlayer.innerHTML = `
        <div class="mp-info">
            <span class="mp-title">${music ? (music.dataset.title || '背景音乐') : '背景音乐'}</span>
            <div class="mp-bar"><div class="mp-bar-progress"></div></div>
        </div>
    `;
    const miniToggle = document.createElement('button');
    miniToggle.className = 'mp-toggle';
    miniToggle.textContent = '▶︎';
    miniPlayer.appendChild(miniToggle);
    document.body.appendChild(miniPlayer);
    if (musicBtn) musicBtn.remove();
    musicBtn = miniToggle;
    const mpProgress = miniPlayer.querySelector('.mp-bar-progress');

    // Function to update highlight.js theme
    function updateHighlightTheme(isDarkMode) {
        let themeLink = document.getElementById('highlight-theme-link');
        if (!themeLink) {
            themeLink = document.createElement('link');
            themeLink.id = 'highlight-theme-link';
            themeLink.rel = 'stylesheet';
            document.head.appendChild(themeLink);
        }
        const lightTheme = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
        const darkTheme = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark-dimmed.min.css';
        themeLink.href = isDarkMode ? darkTheme : lightTheme;
    }

    // Function to update mermaid theme
    function updateMermaidTheme(isDarkMode) {
        if (window.mermaid) {
            mermaid.initialize({
                startOnLoad: true,
                theme: isDarkMode ? 'dark' : 'default',
                themeVariables: isDarkMode ? {
                    primaryColor: '#d6c4f0',
                    primaryTextColor: '#000000',
                    lineColor: '#7a4a93'
                } : {}
            });
            mermaid.init(undefined, document.querySelectorAll('.mermaid'));
        }
    }

    // Initialize highlight.js if present
    function initHighlight() {
        function run() {
            document.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
            if (window.CopyButtonPlugin && hljs.addPlugin) {
                hljs.addPlugin(new CopyButtonPlugin());
            }
        }
        if (window.hljs) {
            run();
        } else {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
            script.onload = () => {
                const plugin = document.createElement('script');
                plugin.src = 'https://cdn.jsdelivr.net/npm/highlightjs-copy@1.0.6/dist/highlightjs-copy.min.js';
                plugin.onload = run;
                document.head.appendChild(plugin);
            };
            document.head.appendChild(script);
        }
    }

    // Function to apply the saved theme
    function applyTheme() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        document.body.classList.toggle('dark-mode', isDarkMode);
        if (darkModeToggle) darkModeToggle.checked = isDarkMode;
        updateHighlightTheme(isDarkMode);
        updateMermaidTheme(isDarkMode);
        initHighlight();
    }

    // Toggle dark mode
    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', () => {
            const isDarkMode = darkModeToggle.checked;
            document.body.classList.toggle('dark-mode', isDarkMode);
            localStorage.setItem('darkMode', isDarkMode);
            updateHighlightTheme(isDarkMode);
            updateMermaidTheme(isDarkMode);
            initHighlight();
            if (darkToggleLabel) {
                darkToggleLabel.classList.add('active');
                setTimeout(() => darkToggleLabel.classList.remove('active'), 300);
            }
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
                    musicBtn.textContent = '⏸';
                }).catch(() => {
                    musicBtn.textContent = '▶︎';
                });
            }
        } else {
            musicBtn.textContent = '▶︎';
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
            musicBtn.textContent = '⏸';
        } else {
            music.pause();
            musicBtn.textContent = '▶︎';
        }
        localStorage.setItem('music-paused', music.paused);
    });

    music.addEventListener('timeupdate', () => {
        localStorage.setItem('music-current-time', music.currentTime);
        if (music.duration) {
            mpProgress.style.width = (music.currentTime / music.duration * 100) + '%';
        }
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

    // Parallax effect for elements with class 'parallax'
    const parallaxEls = document.querySelectorAll('.parallax');
    window.addEventListener('scroll', () => {
        const sy = window.scrollY;
        parallaxEls.forEach(el => {
            const speed = parseFloat(el.dataset.speed || '0.3');
            el.style.transform = `translateY(${sy * speed}px)`;
        });
    });
    
    // Dock 图标放大效果
    const dock = document.querySelector('.dock');
    if (dock) {
        const icons = Array.from(dock.querySelectorAll('a'));
        const current = icons.map(() => 1);
        let targets = icons.map(() => 1);

        dock.addEventListener('mousemove', (e) => {
            targets = icons.map(icon => {
                const rect = icon.getBoundingClientRect();
                const distance = Math.abs(e.clientX - (rect.left + rect.width / 2));
                return Math.max(1, 2 - distance / 100);
            });
        });
        dock.addEventListener('mouseleave', () => {
            targets = icons.map(() => 1);
        });

        function animateDock() {
            icons.forEach((icon, i) => {
                current[i] += (targets[i] - current[i]) * 0.2;
                icon.style.transform = `scale(${current[i]})`;
            });
            requestAnimationFrame(animateDock);
        }
        animateDock();
    }

    // 3D 卡片倾斜与光斑效果
    const tiltCards = document.querySelectorAll('.tilt-card');
    tiltCards.forEach(card => {
        const layers = card.querySelectorAll('[data-depth]');
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -10;
            const rotateY = ((x - centerX) / centerX) * 10;
            card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
            card.style.boxShadow = `${-rotateY}px ${rotateX}px 20px rgba(0,0,0,0.2)`;
            card.style.setProperty('--lx', x + 'px');
            card.style.setProperty('--ly', y + 'px');
            layers.forEach(layer => {
                const depth = parseFloat(layer.dataset.depth || '0');
                const moveX = ((x - centerX) / centerX) * -depth;
                const moveY = ((y - centerY) / centerY) * -depth;
                layer.style.transform = `translate(${moveX}px, ${moveY}px)`;
            });
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'rotateX(0deg) rotateY(0deg)';
            card.style.boxShadow = 'none';
            layers.forEach(layer => {
                layer.style.transform = 'translate(0, 0)';
            });
        });
    });

    // 顶栏与 Dock 随滚动隐藏/显示
    const header = document.querySelector('header');
    let lastY = window.scrollY;
    window.addEventListener('scroll', () => {
        const cur = window.scrollY;
        if (cur > lastY) {
            header && header.classList.add('hidden');
            dock && dock.classList.add('hidden');
        } else {
            header && header.classList.remove('hidden');
            dock && dock.classList.remove('hidden');
        }
        lastY = cur;
    });
    
    // 初始化侧栏功能（在 loadSidebar 完成后调用）
    function initSidebarFeatures() {
        updateArticleTitle();
        generateTOC(); // Add this call
        window.addEventListener('scroll', updateReadingProgress);
        updateReadingProgress();
    }

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
    const ring = document.querySelector('.ring-progress');
    const text = document.querySelector('.progress-ring-text');

    if (!ring || !text) {
        console.log("Could not find progress ring element.");
        return;
    }

    const doc = document.documentElement;
    const scrollableHeight = doc.scrollHeight - doc.clientHeight;
    if (scrollableHeight <= 0) {
        ring.style.strokeDashoffset = 100;
        text.textContent = '0%';
        return;
    }

    const scrollTop = doc.scrollTop || document.body.scrollTop;
    const progressPercentage = (scrollTop / scrollableHeight) * 100;
    ring.style.strokeDashoffset = 100 - progressPercentage;
    text.textContent = Math.round(progressPercentage) + '%';
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

    // Cover Flow 初始化
    async function initCoverFlow() {
        const container = document.getElementById('coverflow');
        if (!container) return;
        let posts;
        try {
            const res = await fetch('posts.json');
            posts = await res.json();
        } catch (e) {
            posts = window.postsData || [];
        }
        const featured = posts.slice(0, 5);
        featured.forEach((post, i) => {
            const item = document.createElement('div');
            item.className = 'coverflow-item';
            item.style.backgroundImage = `url('images/image${String(i+1).padStart(3,'0')}.png')`;
            item.title = post.title;
            container.appendChild(item);
        });
        const items = Array.from(container.children);
        let index = 0;
        function update() {
            items.forEach((el, i) => {
                const offset = i - index;
                el.style.transform = `translateX(${offset * 60}%) translateZ(${ -Math.abs(offset) * 100 }px) rotateY(${offset * -45}deg)`;
                el.style.zIndex = 100 - Math.abs(offset);
            });
        }
        update();
        setInterval(() => {
            index = (index + 1) % items.length;
            update();
        }, 3000);
    }
    initCoverFlow();

    // AirPods 欢迎卡片和特效初始化
    if (location.pathname.endsWith('index.html') || location.pathname === '/') {
        showAirpodsCard();
    }
    initSiriWave();
    initPeekPop();
    initHelloAnimation();

    // VisionOS 漂浮卡片光斑
    document.querySelectorAll('.tilt-card').forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mx', x + 'px');
            card.style.setProperty('--my', y + 'px');
        });
        card.addEventListener('mouseleave', () => {
            card.style.removeProperty('--mx');
            card.style.removeProperty('--my');
        });
    });

    // Taptic Pulse 触感反馈
    function tapticFeedback(e) {
        const el = e.currentTarget;
        el.style.transform = 'scale(0.95)';
        setTimeout(() => { el.style.transform = ''; }, 100);
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = 180;
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
        osc.connect(gain).connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.1);
    }
    document.querySelectorAll('button, .btn, .dock a').forEach(el => {
        el.addEventListener('click', tapticFeedback);
    });
};

function showAirpodsCard() {
    const card = document.createElement('div');
    card.id = 'airpods-card';
    card.innerHTML = `
        <button class="close" aria-label="关闭">×</button>
        <h3>修复了代码高亮的效果</h3>
    `;
    document.body.appendChild(card);
    requestAnimationFrame(() => card.classList.add('show'));
    const remove = () => {
        card.classList.remove('show');
        card.addEventListener('transitionend', () => card.remove(), { once: true });
    };
    card.querySelector('.close').addEventListener('click', remove);
    setTimeout(remove, 5000);
}

function initHelloAnimation() {
    const container = document.getElementById('hello-container');
    if (!container) return;
    const text = container.querySelector('text');
    if (text) {
        text.addEventListener('animationend', () => {
            container.classList.add('fade-out');
            setTimeout(() => {
                container.remove();
                document.body.classList.add('hello-done');
            }, 500);
        }, { once: true });
    }
}

function initSiriWave() {
    document.querySelectorAll('#search-input, #search-input-tech').forEach(input => {
        const container = input.parentElement.querySelector('.siri-wave');
        if (!container) return;
        const canvas = document.createElement('canvas');
        canvas.width = 60; canvas.height = 20;
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d');
        let phase = 0, amp = 0, running = false;
        function draw() {
            if (!running) return;
            ctx.clearRect(0,0,canvas.width,canvas.height);
            const colors = ['#007aff','#5856d6','#5ac8fa'];
            for (let i=0;i<3;i++) {
                ctx.beginPath();
                ctx.strokeStyle = colors[i];
                const offset = i*Math.PI/3;
                for (let x=0;x<canvas.width;x++) {
                    const y = canvas.height/2 + Math.sin((x/canvas.width*2*Math.PI)+phase+offset)*amp;
                    if (x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
                }
                ctx.stroke();
            }
            phase += 0.1;
            requestAnimationFrame(draw);
        }
        function start(){ running = true; container.style.display = 'block'; draw(); }
        function stop(){ running = false; container.style.display = 'none'; amp = 0; }
        input.addEventListener('focus', start);
        input.addEventListener('blur', stop);
        input.addEventListener('input', e => { amp = Math.min(8, 2 + e.target.value.length/2); });
    });
}

async function initPeekPop() {
    const links = document.querySelectorAll('a.post-link');
    if (!links.length) return;
    let posts;
    try {
        const res = await fetch('posts.json');
        posts = await res.json();
    } catch (e) {
        posts = window.postsData || [];
    }
    let timer, card;
    links.forEach(link => {
        link.addEventListener('mousedown', start);
        link.addEventListener('touchstart', start);
        ['mouseup','mouseleave','touchend','touchcancel'].forEach(ev => link.addEventListener(ev, end));

        function start(e) {
            timer = setTimeout(() => {
                const href = link.getAttribute('href');
                const post = posts.find(p => href.endsWith(p.path));
                card = document.createElement('div');
                card.className = 'peek-card';
                card.innerHTML = `<h4>${post ? post.title : ''}</h4><p>${post ? post.abstract : ''}</p>`;
                document.body.appendChild(card);
                requestAnimationFrame(() => card.classList.add('show'));
            }, 300);
        }

        function end(e) {
            clearTimeout(timer);
            if (card) {
                e.preventDefault();
                const href = link.getAttribute('href');
                card.classList.add('pop');
                card.addEventListener('transitionend', () => { window.location.href = href; }, { once:true });
            }
        }
    });
}

function showScreenshotFeedback() {
    document.body.classList.add('flash');
    setTimeout(() => document.body.classList.remove('flash'), 300);
    const img = document.createElement('img');
    img.className = 'screenshot-thumb';
    document.body.appendChild(img);
    const svgData = new XMLSerializer().serializeToString(document.documentElement);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${window.innerWidth}" height="${window.innerHeight}"><foreignObject width="100%" height="100%">${svgData}</foreignObject></svg>`;
    const blob = new Blob([svg], {type:'image/svg+xml;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
        const canvas = document.createElement('canvas');
        const w = 120, h = 80;
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, w, h);
        img.src = canvas.toDataURL();
        URL.revokeObjectURL(url);
        requestAnimationFrame(() => img.classList.add('show'));
        setTimeout(() => {
            img.classList.add('hide');
            img.addEventListener('transitionend', () => img.remove(), {once:true});
        }, 2000);
    };
    image.src = url;
}
document.addEventListener('copy', showScreenshotFeedback);

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