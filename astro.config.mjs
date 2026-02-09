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
// https://astro.build/config
export default defineConfig({
    site: 'https://forliage.github.io',
    base: '/',
    markdown: {
        remarkPlugins: [
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
