# 避坑、好物、小知识合集



# 未解决













## vscode

1. vscode使用正则查找替换，替换的内容前要加$

2. 飞书和markdown的latex格式是不同的，可以通过以下方式将飞书转成markdown
    `\$+$ 替换 $`
    `^\$+$ 替换 $$$`

3. vscode中的jupyter如果file selecter时屏蔽了当前环境，就无法再选择任何环境

4. Office Viewer(Markdown Editor)插件bug很多如选择复制粘贴不替换，Latex内容自动瞎格式化，只能查找无法替换，还是建议使用Markdown All in One插件

5. edamagit插件无法识别jupyter格式内容进行提交，vscode原生则没有

   ```
   GitError! fatal: pathspec '"\347\273\217\345\205\270\347\245\236\347\27 [ $ for detailed log ]
   Head:     main change
   Merge:    origin/main change
   ```

6. vscode能不能屏蔽插件更新提醒的徽章——不能

7. vsode insiders版本无法安装leap插件，报错信息: code-insiders --install-extension vim-1.25.2.vsix --force Installing extensions... (node:95806) [DEP0005] DeprecationWarning: Buffer() is deprecated due to security and usability issues. Please use the Buffer.alloc(), Buffer.allocUnsafe(), or Buffer.from() methods instead. (Use `Electron --trace-deprecation ...` to show where the warning was created) Extension 'vim-1.25.2.vsix' was successfully installed.



## macos

系统更新的弹窗不能屏蔽，但设置的更新红点是可以屏蔽的



当我把打开访达的默认位置设定为下载，我又设置了op+e一个下载路径的快捷键，当我按cmd+e打开方法，再按op+e就会失效，所以还是直接通过alfred访问目录名来的方便，但只要点击下载的任意文件加op+e就不会失效了



safair用于打开飞书虽然十分优雅，但是速度太慢了，现在改用min，min的缺点是开始页面光标不在页面上，窗口优势移动不了的bug到现在还没修好（可以使用设置中的显示浏览器标题栏），还有一个页面最多显示5个窗口，还有不支持本地原生的翻译



mac如何卸载原生自带软件如音乐， 股票

command+R， 关闭SIP：csrutil disable，sudo rm -rf xxx，关闭SIP后安装yabai， 但SIP需要一直关闭，关闭SIP后，在mac上将不能使用Apple Pay开启SIP: csrutil enable

卸载：音乐， 股票，news，播客



mac 无法实现火绒的窗口屏蔽， 对于又有软件更新提醒的APP不太友好



macOS 隐藏正在运行程序在Dock中的图标snippetsLab？





## macapp

1. typora的mac版当图床图片过量（约50）时软件会卡到不能翻页，win版则没有这个问题
1. utools会强制更新，只能通过TripMode来屏蔽其网络连接，淘宝59元，网上破解版都是不能用的。



## typora

1. typora无法通过键盘选择大纲内容进行跳转



## vscode



github提交次数太多，如何删除一些？



如何删除所有的提交记录的所有痕迹?



github如何提交一部分其他人可见，一部分只有自己可见



## 飞书

飞书修改的内容无法与本地同步， 智能手动同步







## Material for MkDocs

无法加载音乐











# 解决



## macos

1. airpos无法多设备流畅切换——airbuddy设置快捷键来解决。



## typora



1. typora如何对所有飞书图片进行重新上传——显示-上传图片









## github



如何查看当前分支所有提交者及其提交次数，按次数由高到低排序？

git log | grep "^Author: " | awk '{print $2}' | sort | uniq -c | sort -k1,1nr



如何查看2023年6月的git提交数目

git log --author=jarrycf --since="2023-06-01" --no-merges | grep -e 'commit [a-zA-Z0-9]*' | wc -l



如何查看对应用户的代码量

git log --author="jarrycf" --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 - $2 } END { printf "added lines: %s, removed lines: %s, total lines: %s\n", add, subs, loc }' -



github提交次数的上限是多少?

无



git 大小写修改无法提交怎么办

```python
git config core.ignorecase false
```





# 开发



node如何更新最新版？

```python
node -v
sudo npm cache clean -f
sudo npm install -g n	
sudo n stable
node -v
sudo npm install npm@latest -g
```



mysql输入密码无法进入mysql

[(46条消息) mac终端报错 ERROR 1045 (28000): Access denied for user ‘root‘@‘localhost‘ (using password: YES)_mac error 1045 (28000): access denied for user 'ro_从小白到能飞起来的博客-CSDN博客](https://blog.csdn.net/m0_62012366/article/details/120397253)

[Mac M1 Mysql 的安装与配置_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV16L4y1b75R/?p=8)





vue如何更新最新版
