# TuSimple 2D Lane Annotation Editor

## Introduction

This project is a PyQt5-based TuSimple 2D lane annotation tool, supporting multi-language interface, suitable for both Ubuntu and Windows systems.

## 1. Ubuntu Development Environment Setup

### 1. Install Python3 and pip

Ubuntu 20.04+ usually comes with Python3. If not, run:

```bash
sudo apt update
sudo apt install python3.10 python3-pip
```

### 2. Install Python Development Package (for packaging)

```bash
sudo apt install python3.10-dev
```

### 3. Install Dependencies

It is recommended to use a virtual environment (optional):

```bash
python3.10 -m pip install --upgrade pip
python3.10 -m pip install virtualenv
python3.10 -m virtualenv venv
source venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run in Development Mode

Make sure `res` directory contains `res_cn.json` and `res_en.json`, then run:

```bash
python3 lane_label_tool.py
```

## 3. Packaging Executable with PyInstaller

### 1. Packaging Command

**It is recommended to package the entire `res` directory:**

```bash
pyinstaller --noconfirm --onefile --add-data "res:res" lane_label_tool.py
```

- `--onefile` generates a single executable file
- `--add-data "res:res"` means the `res` directory will be included in the executable

> Note:  
> - On Windows, use `--add-data "res;res"`; on Linux, use `--add-data "res:res"`
> - The packaged executable will be in `dist/lane_label_tool` (or `dist/lane_label_tool.exe`)

### 2. Run the Packaged Program

```bash
cd dist
./lane_label_tool
```

## 4. Usage

TODO

## 5. FAQ

TODO

## 6. Contact

If you have any questions, please submit an issue or contact the author.
