const fs = require('fs');
const path = require('path');

const blogDir = path.join(__dirname, 'src', 'content', 'blog');

if (!fs.existsSync(blogDir)) {
    console.error('Blog directory not found:', blogDir);
    process.exit(1);
}

fs.readdir(blogDir, (err, files) => {
    if (err) return;

    files.forEach(file => {
        if (path.extname(file) === '.md') {
            const filePath = path.join(blogDir, file);
            let content = fs.readFileSync(filePath, 'utf8');
            let originalContent = content;

            // Fix markdown links: ![alt](url)
            content = content.replace(/!\[(.*?)\]\((.*?)\)/g, (match, alt, url) => {
                if (url.includes('images') || url.includes('.png') || url.includes('.jpg')) {
                    // Normalize url
                    let newUrl = url;
                    // Remove leading .. or .
                    newUrl = newUrl.replace(/^(\.\.[\/\\])+/, '/');
                    newUrl = newUrl.replace(/^(\.[\/\\])+/, '/');

                    // Specific fix for images\ path
                    if (newUrl.startsWith('images\\') || newUrl.startsWith('images/')) {
                        newUrl = '/' + newUrl;
                    }
                    if (newUrl.match(/^\/?images/)) {
                        if (!newUrl.startsWith('/')) newUrl = '/' + newUrl;
                    }

                    // Normalize backslashes
                    newUrl = newUrl.replace(/\\/g, '/');

                    return `![${alt}](${newUrl})`;
                }
                return match;
            });

            // Fix html tags: src="url"
            content = content.replace(/src=["'](.*?)["']/g, (match, url) => {
                if (url.includes('images') || url.includes('.png')) {
                    let newUrl = url;
                    newUrl = newUrl.replace(/^(\.\.[\/\\])+/, '/');
                    newUrl = newUrl.replace(/\\/g, '/');
                    if (newUrl.startsWith('images/')) newUrl = '/' + newUrl;

                    return `src="${newUrl}"`;
                }
                return match;
            });

            if (content !== originalContent) {
                fs.writeFileSync(filePath, content, 'utf8');
                console.log(`Refined paths in: ${file}`);
            }
        }
    });
});
