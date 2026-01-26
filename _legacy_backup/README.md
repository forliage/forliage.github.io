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

### 如何添加新的学习笔记专栏

“学习笔记”分类下的专栏（如“ADS课程笔记”）是根据文章标题自动分组的。要添加一个新的专栏（例如，“操作系统学习笔记”），您需要对 `category.html` 文件进行两处简单的修改。

1.  **添加专栏识别规则**

    打开 `category.html` 文件，找到 `renderLearningNotes` 这个 JavaScript 函数。在此函数内部，您会看到一个名为 `groupedByCourse` 的变量，它使用 `reduce` 方法来对文章进行分组。您需要在这个方法中添加一条 `else if` 规则来识别您的新专栏。

    例如，要添加一个“操作系统”专栏，您可以找到如下代码块：

    ```javascript
    // ...
    const lowerTitle = post.title.toLowerCase();
    if (lowerTitle.startsWith('ads')) {
        courseName = 'ADS 课程笔记';
    } else if (lowerTitle.startsWith('cuda')) {
        courseName = 'CUDA 学习笔记';
    } else if (lowerTitle.startsWith('计算机动画')) {
        courseName = '计算机动画学习笔记';
    }
    // ...
    ```

    在此代码块的最后一个 `else if` 之后，添加一个新的 `else if`。假设您所有操作系统的笔记标题都以 `os` 或 `操作系统` 开头，您可以这样写：

    ```javascript
    // ...
    } else if (lowerTitle.startsWith('计算机动画')) {
        courseName = '计算机动画学习笔记';
    } else if (lowerTitle.startsWith('os') || lowerTitle.startsWith('操作系统')) { // <-- 在这里添加新规则
        courseName = '操作系统学习笔记'; // <-- 这是新专栏的显示名称
    }
    // ...
    ```

2.  **添加专栏简介**

    在同一个 `renderLearningNotes` 函数的顶部，有一个名为 `courseIntroductions` 的对象。您需要在这里为您的新专栏添加一条简介。

    找到如下代码块：
    ```javascript
    const courseIntroductions = {
        'ADS 课程笔记': '这里是关于高级数据结构课程的笔记，涵盖了从基础到进阶的各种数据结构和算法分析。',
        'CUDA 学习笔记': '本专栏记录了 CUDA 并行计算的学习过程，包括基本概念、编程模型和性能优化技巧。',
        '计算机动画学习笔记': '探索计算机动画的奥秘，从关键帧、粒子系统到物理模拟，记录了学习与实践的每一步。',
        '其他': '一些尚未分类的学习笔记。'
    };
    ```
    在其中添加您的新专栏名称和简介，例如：
    ```javascript
    const courseIntroductions = {
        'ADS 课程笔记': '...',
        'CUDA 学习笔记': '...',
        '计算机动画学习笔记': '...',
        '操作系统学习笔记': '这里是关于操作系统课程的笔记，记录了从进程管理到内存管理的各种核心概念。', // <-- 在这里添加新简介
        '其他': '...'
    };
    ```

完成这两处修改后，当您添加的博文标题以`os`开头，且分类为`学习笔记`时，它就会自动出现在新的“操作系统学习笔记”专栏下，并带有您设置的简介。