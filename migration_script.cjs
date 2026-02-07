const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');
const TurndownService = require('turndown');

const postsDir = path.join(__dirname, '_legacy_backup', '_posts');
const outputDir = path.join(__dirname, 'src', 'content', 'blog');
const turndownService = new TurndownService({
    headingStyle: 'atx',
    codeBlockStyle: 'fenced'
});

// Configure turndown to ignore certain elements or handle them specificall
turndownService.remove('script');
turndownService.remove('style');
turndownService.remove('.share-buttons');
turndownService.remove('.dock');
turndownService.remove('header');
turndownService.remove('footer');
turndownService.remove('.modal');

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

fs.readdir(postsDir, (err, files) => {
    if (err) {
        console.error('Error reading posts directory:', err);
        return;
    }

    files.forEach(file => {
        if (path.extname(file) === '.html') {
            const filePath = path.join(postsDir, file);
            const content = fs.readFileSync(filePath, 'utf8');
            const $ = cheerio.load(content);

            // Extract date from filename (YYYY-MM-DD-Title.html)
            const dateMatch = file.match(/^(\d{4}-\d{2}-\d{2})-(.+)\.html$/);
            let date = new Date().toISOString();
            let slug = file.replace('.html', '');

            if (dateMatch) {
                date = dateMatch[1];
                slug = dateMatch[2];
                // slug might contain spaces or special chars, better to keep the original filename base as slug to match old URLs if possible, 
                // but Astro prefers clean slugs. Let's use the full filename without extension as ID for simplicity in content collections.
                slug = file.replace('.html', '');
            }

            // Extract title
            let title = $('head title').text().trim();
            if (!title) {
                title = $('feature-title').text().trim() || $('h2').first().text().trim() || slug;
            }

            // Extract article content
            // The legacy posts seem to have content inside <article> or <main>
            let articleContent = $('article').html();
            if (!articleContent) {
                articleContent = $('main').html() || $('body').html();
            }

            if (articleContent) {
                const markdown = turndownService.turndown(articleContent);

                const frontmatter = `---
title: "${title.replace(/"/g, '\\"')}"
description: ""
pubDate: "${date}"
heroImage: ""
---

`;
                const finalContent = frontmatter + markdown;
                const outputPath = path.join(outputDir, `${slug}.md`);

                fs.writeFileSync(outputPath, finalContent);
                console.log(`Converted: ${file} -> ${slug}.md`);
            } else {
                console.warn(`Skipping ${file}: No content found.`);
            }
        }
    });
});
