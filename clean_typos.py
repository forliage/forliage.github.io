import re

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove specific typo characters found in logs
    content = content.replace('鼓', '')
    content = content.replace('榧', '')
    
    # Also clean up potential syntax errors from previous failed edits
    # Fix the >> error in glob if it exists
    content = re.sub(r'import\.meta\.glob<(\{lectures: Array<\{id: number; title: string; intro: string\}\>\})>>\(', r'import.meta.glob<\1>(', content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

clean_file('src/pages/blog/zju/[courseId].astro')
clean_file('src/pages/blog/zju/[courseId]/[lectureId].astro')
print("Cleaned!")
