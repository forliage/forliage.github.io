import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { visit } from 'unist-util-visit';

/**
 * Custom remark plugin to normalize math delimiters:
 * $$$$ -> $$ (Standard block math, ensure displayMode)
 * $$ -> $ (Standard inline math, ONLY if it doesn't contain matrices)
 */
function remarkMathNormalization() {
    return (tree) => {
        visit(tree, 'text', (node) => {
            if (node.value.includes('$$')) {
                // 1. If it has $$$$, it's definitely block math. 
                // We convert to $$ and add \displaystyle to ensure it looks "big".
                if (node.value.includes('$$$$')) {
                    node.value = node.value.replace(/\$\$\$\$(.*?)\$\$\$\$/gs, (match, p1) => {
                        return `$$\\displaystyle ${p1.trim()}$$`;
                    });
                }

                // 2. Handle $$ (the user wants this to be "formula" but not necessarily centered/big)
                // BUT if it contains matrices, we MUST keep it as block math or it will squash.
                node.value = node.value.replace(/\$\$(.*?)\$\$/gs, (match, p1) => {
                    const content = p1.trim();
                    // Detect common matrix/block environments
                    const isComplex = /\\begin\{|\\matrix|\\bmatrix|\\vmatrix|\\cases|\\aligned/.test(content);

                    if (isComplex) {
                        return `$$${content}$$`; // Keep as block
                    } else {
                        return `$${content}$`; // Convert to inline as requested
                    }
                });
            }
        });
    };
}

// https://astro.build/config
export default defineConfig({
    site: 'https://forliage.github.io',
    base: '/',
    markdown: {
        remarkPlugins: [
            remarkMathNormalization,
            remarkMath
        ],
        rehypePlugins: [
            rehypeRaw,
            [rehypeKatex, { output: 'html', strict: false }]
        ],
    },
});
