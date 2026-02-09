import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { visit } from 'unist-util-visit';

/**
 * Custom remark plugin to normalize math delimiters:
 * $$$$ -> $$ (Standard block/display math)
 * $$ -> $ (Standard inline math)
 */
function remarkMathNormalization() {
    return (tree) => {
        visit(tree, 'text', (node) => {
            if (node.value.includes('$$')) {
                // Step 1: Use placeholders to prevent double-replacement
                // Replace $$$$ with ___BLOCK___
                node.value = node.value.replace(/\$\$\$\$([\s\S]*?)\$\$\$\$/g, (match, content) => {
                    return `___BLOCK_START___${content}___BLOCK_END___`;
                });

                // Step 2: Replace remaining $$ with $ (inline marker)
                node.value = node.value.replace(/\$\$([\s\S]*?)\$\$/g, (match, content) => {
                    return `$${content}$`;
                });

                // Step 3: Restore ___BLOCK___ to $$ (display marker)
                // Add \displaystyle for consistency in display math
                node.value = node.value
                    .replace(/___BLOCK_START___/g, '$$\\displaystyle ')
                    .replace(/___BLOCK_END___/g, '$$');
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
        remarkRehype: {
            allowDangerousHtml: true
        }
    },
});
