# TuSimple 2D 车道线标注工具

## 简介

本项目是一个基于 PyQt5 的 TuSimple 2D 车道线标注工具，支持多语言界面，适用于 Ubuntu 和 Windows 系统。

## 一、Ubuntu 开发环境设置

### 1. 安装 Python3 及 pip

Ubuntu 20.04+ 默认自带 Python3，如未安装可执行：

```bash
sudo apt update
sudo apt install python3.10 python3-pip
```

### 2. 安装 Python 开发包（用于打包）

```bash
sudo apt install python3.10-dev
```

### 3. 安装依赖库

建议使用虚拟环境（可选）：

```bash
python3.10 -m pip install --upgrade pip
python3.10 -m pip install virtualenv
python3.10 -m virtualenv venv
source venv/bin/activate
```

安装项目依赖：

```bash
pip install -r requirements.txt
```
## 二、运行开发版

确保 `res` 目录下有 `res_cn.json` 和 `res_en.json`，然后直接运行：

```bash
python3 lane_label_tool.py
```

## 三、用 PyInstaller 打包可执行文件

### 1. 打包命令

**推荐直接打包整个 res 目录：**

```bash
pyinstaller --noconfirm --onefile --add-data "res:res" lane_label_tool.py
```

- `--onefile` 生成单一可执行文件
- `--add-data "res:res"` 表示将 `res` 目录打包到可执行文件中

> 注意：  
> - Windows 下 `--add-data "res;res"`，Linux 下 `--add-data "res:res"`
> - 打包后可执行文件在 `dist/lane_label_tool`（或 `dist/lane_label_tool.exe`）

### 3. 运行打包后的程序

```bash
cd dist
./lane_label_tool
```

## 四、使用方法

TODO

## 五、常见问题

TODO


## 六、联系方式

如有问题请提交 issue 或联系作者。



