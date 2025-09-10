import os
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
POSTS_DIR = '_posts'
SOCIAL_PLATFORMS = {
    'weibo': {'name': '微博', 'icon': 'sinaweibo'},
    'twitter': {'name': 'Twitter', 'icon': 'twitter'},
    'linkedin': {'name': 'LinkedIn', 'icon': 'linkedin'},
    'wechat': {'name': '微信', 'icon': 'wechat'},
    'qq': {'name': 'QQ', 'icon': 'tencentqq'},
    'facebook': {'name': 'Facebook', 'icon': 'facebook'}
}
CDN_BASE_URL = 'https://cdn.jsdelivr.net/npm/simple-icons/icons/'

# --- HTML Templates ---
DARK_MODE_BUTTON_HTML = '<button id="dark-mode-toggle" class="dark-mode-control">🌙</button>'

HIGHLIGHT_JS_HEAD_HTML = """
<!-- Highlight.js Themes -->
<link id="highlight-theme-link" rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/rose-pine-dawn.min.css">
<!-- Highlight.js Copy Plugin CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlightjs-copy@1.0.6/dist/highlightjs-copy.min.css">
"""

HIGHLIGHT_JS_BODY_HTML = """
<!-- Highlight.js Core -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<!-- Highlight.js Copy Plugin -->
<script src="https://cdn.jsdelivr.net/npm/highlightjs-copy@1.0.6/dist/highlightjs-copy.min.js"></script>
<!-- Initialize Highlight.js and Copy Plugin -->
<script>
  hljs.highlightAll();
  hljs.addPlugin(new CopyButtonPlugin());
</script>
"""

GISCUS_HTML = """
<div class="giscus-container" style="margin-top: 50px;">
  <script src="https://giscus.app/client.js"
          data-repo="forliage/forliage.github.io"
          data-repo-id="R_kgDONjzd4w"
          data-category="Announcements"
          data-mapping="pathname"
          data-strict="0"
          data-reactions-enabled="1"
          data-emit-metadata="0"
          data-input-position="bottom"
          data-theme="https://forliage.github.io/giscus.css"
          data-lang="zh-CN"
          crossorigin="anonymous"
          async>
  </script>
</div>
"""

def get_svg_icon(icon_name):
    """Fetches an SVG icon from the CDN."""
    try:
        url = f"{CDN_BASE_URL}{icon_name}.svg"
        response = requests.get(url)
        if response.status_code == 200:
            svg_content = response.text
            svg_soup = BeautifulSoup(svg_content, 'html.parser')
            svg_tag = svg_soup.find('svg')
            if svg_tag:
                return str(svg_tag)
        return f"<!-- Icon '{icon_name}' not found -->"
    except Exception as e:
        print(f"Error fetching icon {icon_name}: {e}")
        return f"<!-- Error fetching icon '{icon_name}' -->"

def create_share_buttons_html():
    """Creates the full HTML block for the share buttons."""
    buttons_html = '<div class="share-buttons">\n  <p>分享到：</p>\n'
    for platform, details in SOCIAL_PLATFORMS.items():
        icon_svg = get_svg_icon(details['icon'])
        buttons_html += f'  <a href="#" class="share-btn {platform}" onclick="sharePost(event, \'{platform}\')">\n    {icon_svg}\n    <span>{details["name"]}</span>\n  </a>\n'
    buttons_html += '</div>'
    return buttons_html

def process_html_file(filepath, share_buttons_html):
    """Adds features to a single HTML file."""
    print(f"Processing {filepath}...")
    modified = False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        head = soup.find('head')
        body = soup.find('body')

        # --- 1. Add Highlight.js assets ---
        if head and not head.find('link', id='highlight-theme-link'):
            head.append(BeautifulSoup(HIGHLIGHT_JS_HEAD_HTML, 'html.parser'))
            print(f"  - Added Highlight.js CSS to <head>.")
            modified = True

        if body and not body.find('script', src=lambda s: s and 'highlight.min.js' in s):
            body.append(BeautifulSoup(HIGHLIGHT_JS_BODY_HTML, 'html.parser'))
            print(f"  - Added Highlight.js JS to <body>.")
            modified = True

        # --- 2. Add/Update Giscus Comment Section ---
        # Remove existing giscus container to ensure the script is always up-to-date
        giscus_container = soup.find('div', class_='giscus-container')
        if giscus_container:
            giscus_container.decompose()
            print(f"  - Removed old Giscus container to update.")
            modified = True

        share_buttons_div = soup.find('div', class_='share-buttons')
        # We find share_buttons_div to insert after, and we check that we are in a post page (which should have an article tag)
        if share_buttons_div and soup.find('article'):
            share_buttons_div.insert_after(BeautifulSoup(GISCUS_HTML, 'html.parser'))
            print(f"  - Added/Updated Giscus comment section.")
            # If we added Giscus, we must have intended a modification.
            modified = True
        
        # --- Legacy: Add Dark Mode Toggle Button (for older posts if needed) ---
        music_toggle = soup.find('button', id='music-toggle')
        if music_toggle and not music_toggle.find_next_sibling('button', id='dark-mode-toggle'):
            music_toggle.insert_after(BeautifulSoup(DARK_MODE_BUTTON_HTML, 'html.parser'))
            print(f"  - Added dark mode button.")
            modified = True
        
        # --- Legacy: Add Social Share Buttons (for older posts if needed) ---
        article_tag = soup.find('article')
        if article_tag and not soup.find('div', class_='share-buttons'):
            article_tag.insert_after(BeautifulSoup(share_buttons_html, 'html.parser'))
            print(f"  - Added share buttons.")
            modified = True

        # --- Save the modified HTML ---
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            print(f"  - Saved changes to {filepath}")
        else:
            print(f"  - No changes needed.")
            
    except Exception as e:
        print(f"  - Error processing file: {e}")

import argparse

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Add features to blog post HTML files.")
    parser.add_argument('--start', type=int, default=0, help='Start index of files to process.')
    parser.add_argument('--end', type=int, default=None, help='End index of files to process.')
    args = parser.parse_args()

    print(f"Starting to add features to blog posts from index {args.start} to {args.end}...")
    
    # Generate the share buttons HTML once
    share_buttons_html = create_share_buttons_html()
    
    # Get all HTML files and sort them to ensure consistent order
    all_files = sorted([f for f in os.listdir(POSTS_DIR) if f.endswith('.html')])
    
    # Determine the slice of files to process
    if args.end is None:
        files_to_process = all_files[args.start:]
    else:
        files_to_process = all_files[args.start:args.end]

    print(f"Found {len(all_files)} total files. Processing {len(files_to_process)} files in this batch.")

    # Process the selected batch of HTML files
    for filename in files_to_process:
        filepath = os.path.join(POSTS_DIR, filename)
        process_html_file(filepath, share_buttons_html)
            
    print("\nScript finished for this batch.")

if __name__ == '__main__':
    main()
