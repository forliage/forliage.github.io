import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { visit } from 'unist-util-visit';
import katex from 'katex';

/**
 * Custom remark plugin to handle:
 * 1. Delimiter normalization: $$$$ -> $$ (block), $$ -> $ (inline)
 * 2. Math inside HTML: If math is found inside an 'html' node (e.g., <center>$$$$ formula $$$$</center>),
 *    we manually render it so rehype-raw/rehype-katex don't miss it.
 */
function remarkMathAndHtmlProcessing() {
    return (tree) => {
        visit(tree, ['text', 'html'], (node) => {
            if (!node.value || !node.value.includes('$$')) return;

            // Step 1: Handle $$$$ (Block Math)
            // We use a regex to find $$$$ ... $$$$ and replace it.
            // In 'text' nodes, we convert to $$ for remark-math.
            // In 'html' nodes, we must render it immediately because remark-math will skip html nodes.
            if (node.type === 'text') {
                node.value = node.value
                    .replace(/\$\$\$\$([\s\S]*?)\$\$\$\$/g, '$$$1$$') // $$$$ -> $$
                    .replace(/\$\$([\s\S]*?)\$\$/g, '$$1$'); // Remaining $$ -> $
            } else if (node.type === 'html') {
                // Manually render Katex for any math found inside raw HTML tags
                node.value = node.value.replace(/\$\$\$\$([\s\S]*?)\$\$\$\$/g, (match, content) => {
                    return katex.renderToString(content.trim(), { displayMode: true, throwOnError: false });
                });
                node.value = node.value.replace(/\$\$([\s\S]*?)\$\$/g, (match, content) => {
                    return katex.renderToString(content.trim(), { displayMode: false, throwOnError: false });
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
            remarkMathAndHtmlProcessing,
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
