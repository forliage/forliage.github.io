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
                // Replace $$$$ with $$ (block) and $$ with $ (inline)
                // Use a non-greedy regex to match the inner content
                node.value = node.value.replace(/(\${2,4})([\s\S]*?)\1/g, (match, delimiter, content) => {
                    if (delimiter.length === 4) {
                        return `$$${content}$$`;
                    } else if (delimiter.length === 2) {
                        return `$${content}$`;
                    }
                    return match;
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
        remarkRehype: {
            allowDangerousHtml: true
        }
    },
});
