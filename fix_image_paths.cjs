const fs = require('fs');
const path = require('path');

const blogDir = path.join(__dirname, 'src', 'content', 'blog');

if (!fs.existsSync(blogDir)) {
    console.error('Blog directory not found:', blogDir);
    process.exit(1);
}

fs.readdir(blogDir, (err, files) => {
    if (err) {
        console.error('Error reading blog directory:', err);
        return;
    }

    files.forEach(file => {
        if (path.extname(file) === '.md') {
            const filePath = path.join(blogDir, file);
            let content = fs.readFileSync(filePath, 'utf8');
            let originalContent = content;

            // Normalize all backslashes to forward slashes in likely paths first
            // This helps with regex complexity
            // But be careful not to break text? Markdown usually works fine with /.

            // Regex to match ..\images\ or ../images/ or ..\images/ or ../images\
            // AND also just images\

            // Replace ..\images\ or ../images/ with /images/
            content = content.replace(/\.\.[\\\/]images[\\\/]/g, '/images/');

            // Replace src="images\ or src="images/ with src="/images/
            content = content.replace(/src=["']images[\\\/]/g, 'src="/images/');
            content = content.replace(/\(images[\\\/]/g, '(/images/');

            // Handle ComputerNetwork subfolder if it has specific backslashes like images\ComputerNetwork
            // If the previous replace worked, it should be /images/ComputerNetwork...
            // But check for backslashes in the REST of the path?
            // e.g. /images/ComputerNetwork\foo.png -> /images/ComputerNetwork/foo.png

            // Find all /images/... or /images\... paths and normalize slashes within them
            content = content.replace(/\/images[\\\/][^\)\" \n]+/g, (match) => {
                return match.replace(/\\/g, '/');
            });

            if (content !== originalContent) {
                fs.writeFileSync(filePath, content, 'utf8');
                console.log(`Fixed paths in: ${file}`);
            }
        }
    });
});
