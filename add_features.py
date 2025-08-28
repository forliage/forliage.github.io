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

def get_svg_icon(icon_name):
    """Fetches an SVG icon from the CDN."""
    try:
        url = f"{CDN_BASE_URL}{icon_name}.svg"
        response = requests.get(url)
        if response.status_code == 200:
            # The SVG from simple-icons has role and xmlns attributes we can use directly.
            # We might want to add a class for styling.
            svg_content = response.text
            # Let's use BeautifulSoup to manipulate the SVG and add a class
            svg_soup = BeautifulSoup(svg_content, 'html.parser')
            svg_tag = svg_soup.find('svg')
            if svg_tag:
                 # The CSS already handles the fill color via `currentColor`, so we don't need to set it here.
                 # No specific class needed on the svg tag itself based on current CSS.
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
        # Note: The actual sharing URL will be set by JavaScript later.
        # Using a placeholder '#' for now.
        buttons_html += f'  <a href="#" class="share-btn {platform}" onclick="sharePost(event, \'{platform}\')">\n    {icon_svg}\n    <span>{details["name"]}</span>\n  </a>\n'
    buttons_html += '</div>'
    return buttons_html

def process_html_file(filepath, share_buttons_html):
    """Adds features to a single HTML file."""
    print(f"Processing {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # --- 1. Add Dark Mode Toggle Button ---
        music_toggle = soup.find('button', id='music-toggle')
        if music_toggle and not music_toggle.find_next_sibling('button', id='dark-mode-toggle'):
            music_toggle.insert_after(BeautifulSoup(DARK_MODE_BUTTON_HTML, 'html.parser'))
            print(f"  - Added dark mode button.")
        
        # --- 2. Add Social Share Buttons ---
        article_tag = soup.find('article')
        if article_tag and not soup.find('div', class_='share-buttons'):
            article_tag.insert_after(BeautifulSoup(share_buttons_html, 'html.parser'))
            print(f"  - Added share buttons.")

        # --- Save the modified HTML ---
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(soup))
            
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
