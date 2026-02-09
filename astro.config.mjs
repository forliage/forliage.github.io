import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { visit } from 'unist-util-visit';

/**
 * Custom remark plugin to normalize math delimiters:
 * $$$$ -> $$ (Standard block math)
 * $$ -> $ (Standard inline math)
 */
function remarkMathNormalization() {
    return (tree) => {
        visit(tree, 'text', (node) => {
            // Order matters: replace $$$$ first
            if (node.value.includes('$$$$')) {
                node.value = node.value.replace(/\$\$\$\$/g, '$$');
            }
            // Then replace $$ with $ (only if it wasn't part of a $$$$)
            // We use a regex that matches exactly $$ but not $$$$ or $$$
            // However, after the first replace, we've already cleaned up $$$$.
            // A simple loop-based or robust regex is better to avoid double-replacing.

            // Better strategy: replace all $$$$ with a placeholder, then $$ with $, then placeholder with $$
            node.value = node.value
                .replace(/\$\$\$\$/g, '___BLOCK_MATH___')
                .replace(/\$\$/g, '$')
                .replace(/___BLOCK_MATH___/g, '$$');
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
            [rehypeKatex, { output: 'html' }]
        ],
    },
});
