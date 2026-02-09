
const testStr = "$$$$\\mathbf{R}(90^\\circ)=\\begin{bmatrix}0&-1&0\\\\1&0&0\\\\0&0&1\\end{bmatrix}$$$$";

function normalize(val) {
    if (val.includes('$$')) {
        // Step 1: Use placeholders for 4-dollar (block)
        val = val.replace(/\$\$\$\$([\s\S]*?)\$\$\$\$/g, (match, content) => {
            return `___BLOCK_START___${content}___BLOCK_END___`;
        });

        // Step 2: Replace remaining 2-dollar (inline) with 1-dollar
        val = val.replace(/\$\$([\s\S]*?)\$\$/g, (match, content) => {
            return `$${content}$`;
        });

        // Step 3: Restore placeholders to 2-dollar (block)
        val = val.replace(/___BLOCK_START___/g, '$$$') // Use $$$ as temporary to avoid double replacement? No.
            .replace(/___BLOCK_END___/g, '$$$');

        // Wait, if I replace back to $$, Step 2 won't run again anyway.
        val = val.replace(/___BLOCK_START___/g, '$$')
            .replace(/___BLOCK_END___/g, '$$');
    }
    return val;
}

console.log("Input: ", testStr);
console.log("Output:", normalize(testStr));

const testStr2 = "$$inline$$";
console.log("Input: ", testStr2);
console.log("Output:", normalize(testStr2));
