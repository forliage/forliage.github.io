window.onload = function() {
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
    
    // Dock 图标放大效果
    const dock = document.querySelector('.dock');
    if (dock) {
        const icons = dock.querySelectorAll('a');
        dock.addEventListener('mousemove', (e) => {
            icons.forEach(icon => {
                const rect = icon.getBoundingClientRect();
                const distance = Math.abs(e.clientX - (rect.left + rect.width / 2));
                const scale = Math.max(1, 2 - distance / 100);
                icon.style.transform = `scale(${scale})`;
            });
        });
        dock.addEventListener('mouseleave', () => {
            icons.forEach(icon => icon.style.transform = 'scale(1)');
        });
    }

    // 3D 卡片倾斜效果
    const tiltCards = document.querySelectorAll('.tilt-card');
    tiltCards.forEach(card => {
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
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'rotateX(0deg) rotateY(0deg)';
            card.style.boxShadow = 'none';
        });
    });

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