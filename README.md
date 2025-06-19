This repository is for Mr.forliage's blog

Please come to forliage.github.io !

## 使用说明

1. 所有文章放在`_posts`目录下，文件名建议使用`YYYY-MM-DD-標題.html`的格式。
2. 每篇文章开头可以加入YAML前言以标记标题、日期和分类，例如：

   ```
   ---
   title: 我的第一篇博客
   date: 2024-05-01
   category: 心情日记
   ---
   ```

3. YAML前言之后就可以写普通的HTML内容。
4. 添加文章后，在根目录的`posts.json`中记录文章的信息，格式如下：

   ```json
   [
     {
       "title": "我的第一篇博客",
       "date": "2025-06-16",
       "category": "心情日记",
       "path": "_posts/2025-06-16-Welcome.html",
       "abstract": "这是我用HTML撰写的第一篇博客文章，属于\"心情日记\"分类。"
     }
   ]
   ```

   新增文章时，只需在数组末尾追加类似对象即可，`category`字段即为分类名称。若需要离线浏览，可在 `posts.js` 中同步更新相同的数据以供备用。
5. 目前首页展示的是首篇心情日记，其余文章可根据需要扩展导航或索引。