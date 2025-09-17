document.addEventListener('DOMContentLoaded', () => {
    const pageWrapper = document.createElement('div');
    pageWrapper.className = 'page-wrapper';

    // Select only the elements that should be part of the page content transition
    const elementsToWrap = document.querySelectorAll('.hero, .posts-list, footer, .container, .about-content');
    
    if (elementsToWrap.length > 0) {
        // Insert the wrapper before the first element to be wrapped
        const firstElement = elementsToWrap[0];
        firstElement.parentNode.insertBefore(pageWrapper, firstElement);

        // Move the selected elements inside the wrapper
        elementsToWrap.forEach(el => {
            pageWrapper.appendChild(el);
        });
    } else {
        // Fallback for pages without specific content containers, wrap everything except known fixed elements
        const elementsToMove = [];
        for (const child of document.body.children) {
            if (!child.matches('header, .dock, #mini-player, .dark-mode-control, .modal, script')) {
                elementsToMove.push(child);
            }
        }
        if(elementsToMove.length > 0) {
            elementsToMove[0].parentNode.insertBefore(pageWrapper, elementsToMove[0]);
            elementsToMove.forEach(el => pageWrapper.appendChild(el));
        } else {
             document.body.appendChild(pageWrapper); // Append empty wrapper if nothing else
        }
    }

    const bar = document.createElement('div');
    bar.id = 'charging-bar';
    document.body.appendChild(bar); // Append to body to stay fixed
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

    // Task 4: Mini Player Liquid Ripple
    miniPlayer.addEventListener('mousemove', e => {
        const rect = miniPlayer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        miniPlayer.style.setProperty('--mx', `${x}px`);
        miniPlayer.style.setProperty('--my', `${y}px`);
    });
    miniPlayer.addEventListener('mouseleave', () => {
        miniPlayer.style.setProperty('--mx', `-50%`);
        miniPlayer.style.setProperty('--my', `-50%`);
    });

    document.body.appendChild(miniPlayer);
    if (musicBtn) musicBtn.remove();
    musicBtn = miniToggle;
    const mpProgress = miniPlayer.querySelector('.mp-bar-progress');

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

    function updateMermaidTheme(isDarkMode) {
        if (window.mermaid) {
            mermaid.initialize({
                startOnLoad: true,
                theme: isDarkMode ? 'dark' : 'default',
                themeVariables: isDarkMode ? {
                    primaryColor: '#d6c4f0',
                    primaryTextColor: '#000000',
                    nodeTextColor: '#000000',
                    lineColor: '#7a4a93'
                } : {
                    primaryTextColor: '#000000',
                    nodeTextColor: '#000000'
                }
            });
            mermaid.init(undefined, document.querySelectorAll('.mermaid'));
        }
    }

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

    function applyTheme() {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        document.body.classList.toggle('dark-mode', isDarkMode);
        if (darkModeToggle) darkModeToggle.checked = isDarkMode;
        updateHighlightTheme(isDarkMode);
        updateMermaidTheme(isDarkMode);
        initHighlight();
    }

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

    // Style toggle for dark mode control
    const styleToggleBtn = document.getElementById('style-toggle-btn');
    if (styleToggleBtn && darkToggleLabel) {
        const applyToggleStyle = () => {
            const currentStyle = localStorage.getItem('darkToggleStyle') || 'metal';
            if (currentStyle === 'liquid') {
                darkToggleLabel.classList.add('liquid-style');
            } else {
                darkToggleLabel.classList.remove('liquid-style');
            }
        };

        styleToggleBtn.addEventListener('click', () => {
            const isLiquid = darkToggleLabel.classList.toggle('liquid-style');
            localStorage.setItem('darkToggleStyle', isLiquid ? 'liquid' : 'metal');
        });

        applyToggleStyle(); // Apply style on load
    }

    // Make the mini player draggable
    function makeDraggable(elmnt) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        
        // Load position from localStorage
        const savedPos = localStorage.getItem('miniPlayerPosition');
        if (savedPos) {
            const { top, left } = JSON.parse(savedPos);
            elmnt.style.top = top;
            elmnt.style.left = left;
            elmnt.style.right = 'auto';
            elmnt.style.bottom = 'auto';
        }

        const dragMouseDown = (e) => {
            e = e || window.event;
            e.preventDefault();
            pos3 = e.clientX || e.touches[0].clientX;
            pos4 = e.clientY || e.touches[0].clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
            document.ontouchend = closeDragElement;
            document.ontouchmove = elementDrag;
        };

        elmnt.onmousedown = dragMouseDown;
        elmnt.ontouchstart = dragMouseDown;

        const elementDrag = (e) => {
            e = e || window.event;
            const currentX = e.clientX || e.touches[0].clientX;
            const currentY = e.clientY || e.touches[0].clientY;
            pos1 = pos3 - currentX;
            pos2 = pos4 - currentY;
            pos3 = currentX;
            pos4 = currentY;

            let newTop = elmnt.offsetTop - pos2;
            let newLeft = elmnt.offsetLeft - pos1;

            // Boundary check
            const docWidth = document.documentElement.clientWidth;
            const docHeight = document.documentElement.clientHeight;
            const elmntWidth = elmnt.offsetWidth;
            const elmntHeight = elmnt.offsetHeight;

            if (newLeft < 0) newLeft = 0;
            if (newTop < 0) newTop = 0;
            if (newLeft + elmntWidth > docWidth) newLeft = docWidth - elmntWidth;
            if (newTop + elmntHeight > docHeight) newTop = docHeight - elmntHeight;

            elmnt.style.top = newTop + "px";
            elmnt.style.left = newLeft + "px";
            // Important: unset right and bottom when dragging
            elmnt.style.right = 'auto';
            elmnt.style.bottom = 'auto';
        };

        const closeDragElement = () => {
            document.onmouseup = null;
            document.onmousemove = null;
            document.ontouchend = null;
            document.ontouchmove = null;
            
            // Save position to localStorage
            localStorage.setItem('miniPlayerPosition', JSON.stringify({
                top: elmnt.style.top,
                left: elmnt.style.left
            }));
        };
    }

    if (miniPlayer) {
        makeDraggable(miniPlayer);
    }

    applyTheme();

    const savedTime = parseFloat(localStorage.getItem('music-current-time') || '0');
    const shouldPlay = localStorage.getItem('music-paused') !== 'true';

    function restoreMusic() {
        if (!music) return;
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

    if (music) {
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
    }

    window.addEventListener('beforeunload', () => {
        if(music) {
            localStorage.setItem('music-current-time', music.currentTime);
            localStorage.setItem('music-paused', music.paused);
        }
    });

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden' && music) {
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
    if (taglineEl) {
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
    }

    const parallaxEls = document.querySelectorAll('.parallax');
    window.addEventListener('scroll', () => {
        const sy = window.scrollY;
        parallaxEls.forEach(el => {
            const speed = parseFloat(el.dataset.speed || '0.3');
            el.style.transform = `translateY(${sy * speed}px)`;
        });
    });

    // Task 1: Hero Prism Effect
    const heroElement = document.querySelector('.hero');
    if (heroElement) {
        heroElement.addEventListener('mousemove', e => {
            const rect = heroElement.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const xPercent = (x / rect.width) * 100;
            const yPercent = (y / rect.height) * 100;
            heroElement.style.setProperty('--mx', `${xPercent}%`);
            heroElement.style.setProperty('--my', `${yPercent}%`);
        });
    }
    
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

    const header = document.querySelector('header');
    let lastY = window.scrollY;
    window.addEventListener('scroll', () => {
        const cur = window.scrollY;
        if (cur > lastY && cur > 100) { // Add a threshold
            header && header.classList.add('hidden');
            dock && dock.classList.add('hidden');
        } else {
            header && header.classList.remove('hidden');
            dock && dock.classList.remove('hidden');
        }
        lastY = cur;
    });
    
    function initSidebarFeatures() {
        updateArticleTitle();
        generateTOC();
        window.addEventListener('scroll', updateReadingProgress);
        updateReadingProgress();
    }

    function updateArticleTitle() {
        const titlePlaceholder = document.getElementById('current-article-title-placeholder');
        if (!titlePlaceholder) return;
        let title = document.title;
        const mainArticle = document.querySelector('main article');
        if (mainArticle) {
            const heading = mainArticle.querySelector('h1, h2, h3, h4, h5, h6');
            if (heading) {
                title = heading.textContent;
            }
        }
        titlePlaceholder.textContent = title || '[无法获取标题]';
    }

    function updateReadingProgress() {
        const ring = document.querySelector('.ring-progress');
        const text = document.querySelector('.progress-ring-text');
        if (!ring || !text) return;

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
        if (!tocContainer || !mainArticle) return;

        const headings = mainArticle.querySelectorAll('h1, h2, h3, h4');
        if (headings.length === 0) {
            tocContainer.innerHTML = '<p style="font-size: 0.9em; color: #666;">本文无目录。</p>';
            return;
        }

        const tocList = document.createElement('ul');
        tocList.className = 'toc-list';
        headings.forEach(heading => {
            if (!heading.id) return;
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            listItem.className = `toc-level-${heading.tagName.toLowerCase()}`;
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent;
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetElement = document.getElementById(heading.id);
                if(targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            });
            listItem.appendChild(link);
            tocList.appendChild(listItem);
        });
        tocContainer.appendChild(tocList);
    }

    document.addEventListener('mousedown', function showXinOnClick(event) {
        if (event.button !== 0) return;
        const xinElement = document.createElement('span');
        xinElement.textContent = '♥';
        const fontSize = 24;
        const offsetX = fontSize / 2; 
        const offsetY = fontSize / 2; 
        xinElement.style.position = 'fixed';
        xinElement.style.left = (event.clientX - offsetX) + 'px';
        xinElement.style.top = (event.clientY - offsetY) + 'px';
        xinElement.style.color = 'red';
        xinElement.style.fontSize = fontSize + 'px';
        xinElement.style.pointerEvents = 'none';
        xinElement.style.userSelect = 'none';
        xinElement.style.transition = 'opacity 0.5s ease-out, transform 0.4s ease-out';
        xinElement.style.opacity = '1';
        xinElement.style.transform = 'translateY(0)';
        document.body.appendChild(xinElement);
        setTimeout(() => {
            xinElement.style.opacity = '0';
            xinElement.style.transform = 'translateY(-20px)';
        }, 50);
        setTimeout(() => {
            if (xinElement.parentNode) {
                xinElement.parentNode.removeChild(xinElement);
            }
        }, 550);
    });

    const glassEffectToggle = document.getElementById('glass-effect-toggle');
    const blurSlider = document.getElementById('blur-slider');
    if (glassEffectToggle && blurSlider) {
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
            glassEffectToggle.textContent = isEnabled ? 'Close' : 'Blur';
            blurSlider.style.display = isEnabled ? 'block' : 'none';
        });
        blurSlider.addEventListener('input', (e) => {
            const blurValue = e.target.value;
            document.documentElement.style.setProperty('--blur-intensity', blurValue + 'px');
            localStorage.setItem('blurIntensity', blurValue);
        });
    }

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
    
    if (location.pathname.endsWith('index.html') || location.pathname === '/') {
        showAirpodsCard();
    }
    
    // --- NEW FEATURES INITIALIZATION ---
    initSiriWave();
    initFloatingCards();
    initGestureNavigation();
    initContinueReading();
    // --- END NEW FEATURES INITIALIZATION ---

    initPeekPop();
    initHelloAnimation();

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
        <h3>增加了亮暗切换按钮的液态玻璃效果选项</h3>
        <p style="font-size: 0.8em; margin: 5px 0 0;">大家可以根据自己的喜欢在金属效果与液态玻璃效果间进行切换</p>
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

// --- TASK 3: Siri Wave ---
function initSiriWave() {
    document.querySelectorAll('#search-input, #search-input-tech').forEach(input => {
        const container = input.parentElement.querySelector('.siri-wave');
        if (!container) return;
        container.innerHTML = ''; // Clear previous canvas if any
        const canvas = document.createElement('canvas');
        canvas.width = 120; // Increased width for better look
        canvas.height = 40;
        canvas.style.width = '60px';
        canvas.style.height = '20px';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        let phase = 0, targetAmp = 0, currentAmp = 0, animationFrame;
        let typingTimeout;

        function draw() {
            currentAmp += (targetAmp - currentAmp) * 0.1;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (currentAmp === 0) {
                cancelAnimationFrame(animationFrame);
                return;
            }

            const colors = ['#ff8acb', '#AD1457', '#C2185B'];
            for (let i = 0; i < 3; i++) {
                ctx.beginPath();
                ctx.strokeStyle = colors[i];
                ctx.lineWidth = 1.5;
                const offset = i * Math.PI / 3;
                for (let x = 0; x < canvas.width; x++) {
                    const y = canvas.height / 2 + Math.sin((x / canvas.width * 2 * Math.PI) + phase + offset) * currentAmp;
                    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            phase += 0.1;
            animationFrame = requestAnimationFrame(draw);
        }

        function startWave() {
            if (!animationFrame) {
                animationFrame = requestAnimationFrame(draw);
            }
        }

        input.addEventListener('focus', () => {
            container.style.display = 'block';
            targetAmp = 0.5; // Show a flat line
            startWave();
        });

        input.addEventListener('blur', () => {
            targetAmp = 0;
            setTimeout(() => { 
                if(targetAmp === 0) container.style.display = 'none';
            }, 200);
        });

        input.addEventListener('input', () => {
            clearTimeout(typingTimeout);
            targetAmp = Math.min(15, 2 + input.value.length);
            startWave();
            typingTimeout = setTimeout(() => {
                targetAmp = 0.5; // Return to flat line
            }, 500);
        });
    });
}

// --- TASK 2: Floating Cards ---
function initFloatingCards() {
    const cards = document.querySelectorAll('.course-card');
    if (!cards.length) return;

    // Desktop hover fallback
    cards.forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const rotateX = (y / rect.height - 0.5) * -10; // Subtle rotation
            const rotateY = (x / rect.width - 0.5) * 10;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
            card.style.transition = 'transform 0.1s';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
            card.style.transition = 'transform 0.5s';
        });
    });

    // Mobile device orientation
    if (window.DeviceOrientationEvent) {
        window.addEventListener('deviceorientation', e => {
            if (e.gamma === null || e.beta === null) return;
            const beta = Math.max(-30, Math.min(30, e.beta)); // Front-back tilt
            const gamma = Math.max(-30, Math.min(30, e.gamma)); // Left-right tilt
            
            cards.forEach(card => {
                const rotateX = beta * -0.2; // Subtle effect
                const rotateY = gamma * 0.2;
                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
                card.style.transition = 'transform 0.2s';
            });
        });
    }
}

// --- TASK 1: Gesture Navigation ---
function initGestureNavigation() {
    const applicablePages = ['/index.html', '/about.html', '/category.html', '/'];
    if (!applicablePages.includes(location.pathname)) return;

    let touchStartX = 0;
    let touchStartY = 0;
    let touchEndX = 0;
    let touchEndY = 0;
    const swipeThreshold = window.innerWidth * 0.4; // 40% of screen width

    document.body.addEventListener('touchstart', e => {
        if (e.touches.length === 2) {
            touchStartX = e.touches[0].screenX;
            touchStartY = e.touches[0].screenY;
        }
    }, { passive: true });

    document.body.addEventListener('touchmove', e => {
        if (e.touches.length === 2) {
            touchEndX = e.touches[0].screenX;
            touchEndY = e.touches[0].screenY;
        }
    }, { passive: true });

    document.body.addEventListener('touchend', e => {
        if (e.touches.length === 0 && touchStartX !== 0) {
            const dx = touchEndX - touchStartX;
            const dy = touchEndY - touchStartY;

            if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > swipeThreshold) {
                // It's a horizontal swipe
                const action = dx > 0 ? 'back' : 'forward';
                document.body.classList.add(dx > 0 ? 'is-exiting-back' : 'is-exiting');
                setTimeout(() => {
                    if (action === 'back') history.back();
                    else history.forward();
                }, 500); // Match CSS animation duration
            }
            
            // Reset coordinates
            touchStartX = 0;
            touchStartY = 0;
            touchEndX = 0;
            touchEndY = 0;
        }
    }, { passive: true });
}

// --- TASK 4: Continue Reading ---
function initContinueReading() {
    const isPostPage = location.pathname.includes('/_posts/');
    const isHomePage = location.pathname.endsWith('/index.html') || location.pathname === '/';

    if (isPostPage) {
        // On a post page, track scroll position
        let lastKnownScrollPosition = 0;
        let ticking = false;

        window.addEventListener('scroll', () => {
            lastKnownScrollPosition = window.scrollY;
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    const scrollableHeight = document.documentElement.scrollHeight - window.innerHeight;
                    const scrollPercent = (lastKnownScrollPosition / scrollableHeight) * 100;
                    
                    if (scrollPercent < 95) {
                        const articleTitle = document.title;
                        localStorage.setItem('continueReading', JSON.stringify({
                            url: location.href,
                            title: articleTitle,
                            scrollPos: lastKnownScrollPosition
                        }));
                    } else {
                        // If they finished, remove the prompt for this article
                        const lastRead = JSON.parse(localStorage.getItem('continueReading') || '{}');
                        if (lastRead.url === location.href) {
                            localStorage.removeItem('continueReading');
                        }
                    }
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    if (isHomePage) {
        // On the home page, show the banner if needed
        const lastReadData = localStorage.getItem('continueReading');
        if (lastReadData) {
            const { url, title, scrollPos } = JSON.parse(lastReadData);
            
            const banner = document.createElement('div');
            banner.id = 'continue-reading-banner';
            
            const text = document.createElement('p');
            text.innerHTML = `上次您读到：<strong>${title.split(' - ')[0]}</strong>`;

            const link = document.createElement('a');
            link.className = 'continue-link';
            link.textContent = '继续阅读';
            link.href = '#'; // Prevent page reload
            link.onclick = (e) => {
                e.preventDefault();
                localStorage.setItem('scrollToPos', scrollPos);
                window.location.href = url;
            };

            const closeBtn = document.createElement('button');
            closeBtn.className = 'close-btn';
            closeBtn.innerHTML = '&times;';
            closeBtn.onclick = () => {
                banner.classList.remove('show');
                // Don't permanently delete, just hide for this session
                sessionStorage.setItem('continueReadingDismissed', 'true');
            };

            banner.appendChild(text);
            banner.appendChild(link);
            banner.appendChild(closeBtn);
            document.body.appendChild(banner);

            if (sessionStorage.getItem('continueReadingDismissed') !== 'true') {
                 setTimeout(() => banner.classList.add('show'), 500);
            }
        }
    }
    
    // On page load, check if we need to scroll
    const scrollToPos = localStorage.getItem('scrollToPos');
    if (scrollToPos) {
        window.scrollTo(0, parseInt(scrollToPos, 10));
        localStorage.removeItem('scrollToPos');
    }
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