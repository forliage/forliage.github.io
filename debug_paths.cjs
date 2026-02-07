const fs = require('fs');
const path = require('path');

const blogDir = path.join(__dirname, 'src', 'content', 'blog');

fs.readdir(blogDir, (err, files) => {
    if (err) return;
    files.forEach(file => {
        const filePath = path.join(blogDir, file);
        if (fs.statSync(filePath).isFile()) {
            const content = fs.readFileSync(filePath, 'utf8');
            if (content.includes('../images') || content.includes('..\\images')) {
                console.log(`FOUND RELATIVE PATH IN: ${file}`);
                // Fix it
                let newContent = content.replace(/\.\.\/images/g, '/images');
                newContent = newContent.replace(/\.\.\\images/g, '/images');
                fs.writeFileSync(filePath, newContent);
                console.log(`Fixed ${file}`);
            }
        }
    });
});
