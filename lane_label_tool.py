#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import PyQt5.QtCore
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.dirname(PyQt5.QtCore.__file__) + "/plugins"

import sys
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QListWidget, QMessageBox, QInputDialog, QListWidgetItem, QCheckBox,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QComboBox  # 新增 QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint

LANE_COLORS = [
    QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
    QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)
]

LANE_COLOR_NAMES = ["red", "green", "blue", "yellow", "purple", "cyan"]


TUSIMPLE_IMG_SIZE = (1280, 720)
CANVAS_SIZE = (960, 540)

LANG_EN = "EN"
LANG_CN = "CN"
CFG_LANGS = [LANG_EN, LANG_CN]

# 修改 ConfigDialog 类
class ConfigDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.lang_manager = parent.lang_manager
        self.setWindowTitle(self.lang_manager.get_text("config_title"))
        self.setModal(True)
        self.resize(500, 300)
        
        # 创建表单布局
        layout = QFormLayout()
        
        # 数据集根路径配置
        self.image_root = QLineEdit(self)
        self.image_root.setText(config.get("image_root", "datasets/TUSimple/tusimple"))
        self.image_root.setMinimumWidth(300)
        layout.addRow(self.lang_manager.get_text("config_image_root"), self.image_root)
        
        # 项目ID配置
        self.project_id = QLineEdit(self)
        self.project_id.setText(config.get("project_id", "tusimple_lane"))
        layout.addRow(self.lang_manager.get_text("config_project_id"), self.project_id)
        
        # 最大车道线数配置
        self.max_lanes = QLineEdit(self)
        self.max_lanes.setText(str(config.get("max_lanes", 6)))
        layout.addRow(self.lang_manager.get_text("config_max_lanes"), self.max_lanes)
        
        # 新增：语言选择下拉框
        self.lang_combo = QComboBox(self)
        self.lang_combo.addItems(["中文", "English"])
        current_lang = config.get("lang", "CN")
        self.lang_combo.setCurrentText("中文" if current_lang == "CN" else "English")
        layout.addRow(self.lang_manager.get_text("config_lang"), self.lang_combo)
        
        # 添加确定和取消按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(button_box)
        
        self.setLayout(main_layout)
    
    def get_config(self):
        """获取配置信息"""
        try:
            max_lanes = int(self.max_lanes.text())
        except ValueError:
            max_lanes = 6
            
        return {
            "image_root": self.image_root.text(),
            "project_id": self.project_id.text(),
            "max_lanes": max_lanes,
            "lang": "CN" if self.lang_combo.currentText() == "中文" else "EN"
        }

class LanguageManager:
    def __init__(self):
        self.resources = {}
        self.current_lang = "CN"
        self.load_resources()

    def load_resources(self):
        """加载语言资源文件"""
        try:
            with open("./res/res_cn.json", "r", encoding="utf-8") as f:
                self.resources["CN"] = json.load(f)
            with open("./res/res_en.json", "r", encoding="utf-8") as f:
                self.resources["EN"] = json.load(f)
        except Exception as e:
            print(f"加载语言资源文件失败: {e}")
            self.resources["CN"] = {}
            self.resources["EN"] = {}

    def set_language(self, lang):
        """设置当前语言"""
        if lang in ["CN", "EN"]:
            self.current_lang = lang

    def get_text(self, key, **kwargs):
        """获取指定key的文本，支持格式化参数"""
        text = self.resources.get(self.current_lang, {}).get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except:
                return text
        return text

class LaneLabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()  # 先加载配置
        self.lang_manager = LanguageManager()  # 创建语言管理器
        self.lang_manager.set_language(self.config.get("lang", "CN"))  # 设置语言
        
        self.setWindowTitle(self.lang_manager.get_text("window_title"))
        #self.resize(1200, 800)
        #self.setMinimumSize(1600, 900)
        self.resize(1600, 900)
        self.annotation_data = []
        self.current_index = 0  # 当前标注的图片索引
        self.current_lane = 0  # 当前标注的车道线索引
        self.undo_stack = []  # 撤销栈
        self.redo_stack = []  # 重做栈
        self.image = None
        self.img_h_scale = CANVAS_SIZE[1] / TUSIMPLE_IMG_SIZE[1]
        self.img_w_scale = CANVAS_SIZE[0] / TUSIMPLE_IMG_SIZE[0]
        self.image_path = ""
        self.h_samples = []
        self.lane_points = []  # [[(x1, y1), (x2, y2), ...], ...]
        self.path_label = QLabel("")  # 新增：用于显示路径和分辨率
        self.json_file_label = QLabel("")  # 新增：用于显示json文件名
        self.json_file_label.setAlignment(Qt.AlignLeft)
        self.json_file_path = None
        self.cache = self.load_cache()  # 修改：加载完整的缓存信息
        self.last_json_path = self.cache.get("last_json_path", "")
        self.select_all_checkbox = None  # 新增：全选复选框
        self.selected_lane_indices = set()  # 新增：用于多选支持
        self.last_saved_lane_points = None  # 新增：用于保存上次保存的lane_points快照
        self.dataset_path = "datasets/TUSimple/tusimple"  # 新增：默认数据集路径
        self.project_id_label = QLabel("")  # 新增：用于显示project id
        self.project_id_label.setAlignment(Qt.AlignLeft)
        self.init_ui()

    def init_ui(self):
        # 顶部按钮
        top_layout = QHBoxLayout()
        
        # 左侧按钮组
        left_buttons = QHBoxLayout()
        open_btn = QPushButton(self.lang_manager.get_text("btn_open"))
        open_btn.clicked.connect(self.open_annotation)
        save_copy_btn = QPushButton(self.lang_manager.get_text("btn_save_copy"))
        save_copy_btn.clicked.connect(self.save_copy)
        prev_btn = QPushButton(self.lang_manager.get_text("btn_prev"))
        prev_btn.clicked.connect(self.prev_image)
        next_btn = QPushButton(self.lang_manager.get_text("btn_next"))
        next_btn.clicked.connect(self.next_image)
        
        left_buttons.addWidget(open_btn)
        left_buttons.addWidget(save_copy_btn)  # 移回左侧
        left_buttons.addWidget(prev_btn)
        left_buttons.addWidget(next_btn)
        
        # 右侧按钮组
        right_buttons = QHBoxLayout()
        # 配置按钮
        config_btn = QPushButton("⚙️")  # 使用齿轮emoji作为图标
        config_btn.setFixedSize(30, 30)  # 设置按钮大小
        config_btn.clicked.connect(self.show_config_dialog)
        
        # 保存按钮
        save_btn = QPushButton(self.lang_manager.get_text("btn_save"))
        save_btn.clicked.connect(self.save_annotation)
        
        right_buttons.addWidget(config_btn)
        right_buttons.addWidget(save_btn)
        
        # 将左右按钮组添加到顶部布局
        top_layout.addLayout(left_buttons)
        top_layout.addStretch()  # 添加弹性空间
        top_layout.addLayout(right_buttons)

        # 新增：按钮下方显示project id、json文件名、路径和分辨率
        path_layout = QVBoxLayout()
        path_layout.addLayout(top_layout)
        path_layout.addWidget(self.project_id_label)  # 新增：添加project id label
        path_layout.addWidget(self.json_file_label)
        path_layout.addWidget(self.path_label)

        right_layout = QVBoxLayout()
        self.lane_list = QListWidget()
        self.lane_list.setSelectionMode(QListWidget.SingleSelection)  # 保持单选模式
        self.lane_list.currentRowChanged.connect(self.select_lane)

        self.select_all_checkbox = QCheckBox(self.lang_manager.get_text("checkbox_select_all"))
        self.select_all_checkbox.setChecked(True)
        self.select_all_checkbox.stateChanged.connect(self.on_select_all_changed)

        add_lane_btn = QPushButton(self.lang_manager.get_text("btn_add_lane"))
        add_lane_btn.clicked.connect(self.add_lane)
        del_lane_btn = QPushButton(self.lang_manager.get_text("btn_del_lane"))
        del_lane_btn.clicked.connect(self.delete_lane)
        undo_btn = QPushButton(self.lang_manager.get_text("btn_undo"))
        undo_btn.clicked.connect(self.undo)
        redo_btn = QPushButton(self.lang_manager.get_text("btn_redo"))
        redo_btn.clicked.connect(self.redo)

        show_points_btn = QPushButton(self.lang_manager.get_text("btn_show_points"))
        show_points_btn.clicked.connect(self.show_current_lane_points)

        # 新增：整理当前车道线按钮
        organize_btn = QPushButton(self.lang_manager.get_text("btn_organize"))
        organize_btn.clicked.connect(self.organize_current_lane)

        lane_list_label = QLabel(self.lang_manager.get_text("label_lane_list"))
        right_layout.addWidget(lane_list_label)
        right_layout.addWidget(self.select_all_checkbox)
        right_layout.addWidget(self.lane_list)
        right_layout.addWidget(add_lane_btn)
        right_layout.addWidget(del_lane_btn)
        right_layout.addWidget(undo_btn)
        right_layout.addWidget(redo_btn)
        right_layout.addWidget(show_points_btn)
        right_layout.addWidget(organize_btn)  # 新增：整理按钮
        right_layout.addStretch()

        main_layout = QHBoxLayout()
        self.canvas = QLabel()
        self.canvas.setFixedSize(TUSIMPLE_IMG_SIZE[0], TUSIMPLE_IMG_SIZE[1])
        self.canvas.setMouseTracking(True)
        self.canvas.mousePressEvent = self.on_canvas_click
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(right_layout)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addLayout(path_layout)  # 用 path_layout 替换 top_layout
        layout.addLayout(main_layout)
        self.setCentralWidget(central_widget)

    def load_cache(self):
        """加载缓存信息，包括上次标注的文件路径、文件名和图片索引"""
        cache_file = "cache.json"
        default_cache = {
            "last_json_path": "",      # 上次打开的目录
            "json_file_path": None,    # 上次打开的文件完整路径
            "current_index": 0,        # 上次标注的图片索引
            "dataset_path": "datasets/TUSimple/tusimple"  # 新增：默认数据集路径
        }
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
                    # 加载数据集路径
                    self.dataset_path = cache.get("dataset_path", default_cache["dataset_path"])
                    return cache
            except Exception:
                return default_cache
        return default_cache

    def save_cache(self):
        """保存缓存信息"""
        cache = {
            "last_json_path": self.last_json_path,
            "json_file_path": self.json_file_path,
            "current_index": self.current_index,
            "dataset_path": self.dataset_path  # 新增：保存数据集路径
        }
        with open("cache.json", "w") as f:
            json.dump(cache, f)

    def open_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.lang_manager.get_text("dialog_open_file"), 
            self.last_json_path, "JSON Files (*.json)"
        )
        if not file_path:
            return
        self._open_annotation(file_path)

    def _open_annotation(self, file_path):
        self.last_json_path = os.path.dirname(file_path)
        self.json_file_path = file_path
        self.annotation_data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.annotation_data.append(json.loads(line))
        
        # 如果是打开上次的文件，恢复上次的索引位置
        if file_path == self.cache.get("json_file_path"):
            saved_index = self.cache.get("current_index", 0)
            if 0 <= saved_index < len(self.annotation_data):
                self.current_index = saved_index
            else:
                self.current_index = 0
        else:
            self.current_index = 0
            
        self.project_id_label.setText(
            self.lang_manager.get_text("label_project_id", id=self.config['project_id']))
        self.json_file_label.setText(
            self.lang_manager.get_text("label_json_file", filename=os.path.basename(file_path)))
        self.load_image_and_lanes()
        self.last_saved_lane_points = json.dumps(self.lane_points)
        self.save_cache()  # 保存新的缓存信息

    def save_annotation(self):
        if not self.annotation_data:
            return

        # Check if current lanes exceed max_lanes
        if len(self.lane_points) > self.config["max_lanes"]:
            QMessageBox.warning(
                self,
                self.lang_manager.get_text("dialog_warning"),
                self.lang_manager.get_text("msg_too_many_lanes", 
                    current=len(self.lane_points), 
                    max=self.config["max_lanes"])
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, self.lang_manager.get_text("dialog_save_file"), 
            "", "JSON Files (*.json)")
        if not file_path:
            return
        
        # 新增：保存前自动检查并插值
        self.auto_interpolate_all_lanes_to_h_samples()
        self.save_current_lane_points_to_annotation()

        self.last_json_path = os.path.dirname(file_path)
        self.save_cache()  # 退出前保存缓存
        with open(file_path, "w") as f:
            for ann in self.annotation_data:
                json.dump(ann, f)
                f.write("\n")
        QMessageBox.information(
            self, 
            self.lang_manager.get_text("dialog_success"), 
            self.lang_manager.get_text("msg_save_success")
        )
        self.last_saved_lane_points = json.dumps(self.lane_points)  # 新增：保存后更新快照

    def auto_interpolate_all_lanes_to_h_samples(self):
        """
        检查当前图片的所有lane_points是否都是h_samples上的点，如果不是则自动做线性插值。
        """
        if not self.h_samples or not self.lane_points:
            return
        new_lane_points = []
        for lane in self.lane_points:
            if not lane or len(lane) < 2:
                new_lane_points.append(lane)
                continue
            # 检查是否所有点的y都在h_samples上
            lane_ys = [pt[1] for pt in lane]
            if all(y in self.h_samples for y in lane_ys) and len(lane) == len(self.h_samples):
                new_lane_points.append(lane)
                continue
            # 需要插值
            points = sorted(lane, key=lambda x: x[1])
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            min_y, max_y = min(ys), max(ys)
            interp_h_samples = [y for y in self.h_samples if min_y <= y <= max_y]
            if len(interp_h_samples) == 0:
                print(self.lang_manager.get_text("msg_lane_deleted", 
                    index=lane_idx))
                continue
            interp_xs = np.interp(interp_h_samples, ys, xs)
            new_points = [(int(round(x)), int(y)) for x, y in zip(interp_xs, interp_h_samples)]
            new_lane_points.append(new_points)
        self.lane_points = new_lane_points
        self.update_lane_list()
        self.update_canvas()

    def prev_image(self):
        if self.current_index > 0:
            if not self.check_unsaved_changes():
                return
            self.current_index -= 1
            self.load_image_and_lanes()
            self.last_saved_lane_points = json.dumps(self.lane_points)
            self.save_cache()  # 保存当前索引

    def next_image(self):
        if self.current_index < len(self.annotation_data) - 1:
            if not self.check_unsaved_changes():
                return
            self.current_index += 1
            self.load_image_and_lanes()
            self.last_saved_lane_points = json.dumps(self.lane_points)
            self.save_cache()  # 保存当前索引

    def check_unsaved_changes(self):
        """
        检查当前车道线像素点是否有未保存的更改，有则弹窗提醒用户是否保存。
        返回True表示可以切换，False表示用户取消切换。
        """
        current = json.dumps(self.lane_points)
        if self.last_saved_lane_points is not None and current != self.last_saved_lane_points:
            reply = QMessageBox.question(
                self, "未保存的更改",
                "当前图片的车道线有未保存的更改，是否保存？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                # 保存到annotation_data
                #self.save_annotation()
                self._save_copy()
                return True
            elif reply == QMessageBox.No:
                return True
            else:
                return False
        return True

    def save_current_lane_points_to_annotation(self):
        """
        将当前self.lane_points同步回self.annotation_data[self.current_index]['lanes']。
        """
        if not self.annotation_data:
            return
        # 只保存x坐标，y坐标由h_samples决定
        lanes = []
        for lane in self.lane_points:
            lane_xs = []
            lane_dict = {}
            for y in self.h_samples:
                # 查找与y匹配的点
                found = False
                for pt in lane:
                    if pt[1] == y:
                        lane_xs.append(pt[0])
                        found = True
                        break
                if not found:
                    lane_xs.append(-2)  # 按tusimple格式，未标注点为-2
            lanes.append(lane_xs)
        self.annotation_data[self.current_index]['lanes'] = lanes

    def load_image_and_lanes(self):
        ann = self.annotation_data[self.current_index]
        self.image_path = os.path.join(self.config["image_root"], ann["raw_file"])
        self.h_samples = ann["h_samples"]
        self.lane_points = []
        for lane in ann["lanes"]:
            points = []
            for x, y in zip(lane, self.h_samples):
                if x >= 0:
                    points.append((x, y))
            self.lane_points.append(points)
        self.current_lane = 0
        self.select_all_checkbox.setChecked(True)
        self.update_lane_list()
        self.load_image()
        self.update_canvas()
        # 更新路径和分辨率显示
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.path_label.setText(
                self.lang_manager.get_text("label_image_info", 
                    index=self.current_index,
                    path=self.image_path,
                    width=w,
                    height=h
                )
            )
        else:
            self.path_label.setText(
                self.lang_manager.get_text("label_image_not_loaded",
                    index=self.current_index,
                    path=self.image_path
                )
            )
        

    def load_image(self):
        img = cv2.imread(self.image_path)
        img_h, img_w = img.shape[:2]
        assert img_h == TUSIMPLE_IMG_SIZE[1] and img_w == TUSIMPLE_IMG_SIZE[0], f"Image size mismatch, img_h: {img_h}, img_w: {img_w}"
        if img is None:
            #self.image = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
            self.image = np.zeros((TUSIMPLE_IMG_SIZE[1], TUSIMPLE_IMG_SIZE[0], 3), dtype=np.uint8)
        else:
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #self.image = cv2.resize(self.image, (CANVAS_SIZE[0], CANVAS_SIZE[1]))

    def update_lane_list(self):
        self.lane_list.clear()
        for idx, lane in enumerate(self.lane_points):
            item_text = self.lang_manager.get_text("lane_item", 
                index=idx+1, 
                count=len(lane), 
                color=LANE_COLOR_NAMES[idx % len(LANE_COLORS)])
            item = QListWidgetItem(item_text)
            color = LANE_COLORS[idx % len(LANE_COLORS)]
            item.setForeground(color)
            self.lane_list.addItem(item)
        self.lane_list.setCurrentRow(self.current_lane)

    def select_lane(self, idx):
        if 0 <= idx < len(self.lane_points):
            self.current_lane = idx
            self.update_canvas()

    def add_lane(self):
        # 检查是否达到最大车道线数
        if len(self.lane_points) >= self.config["max_lanes"]:
            QMessageBox.warning(self, "警告", f"已达到最大车道线数量({self.config['max_lanes']})")
            return
            
        self.push_undo()
        self.lane_points.append([])
        self.current_lane = len(self.lane_points) - 1
        self.update_lane_list()
        self.update_canvas()

    def delete_lane(self):
        if len(self.lane_points) == 0:
            return
        self.push_undo()
        del self.lane_points[self.current_lane]
        self.current_lane = max(0, self.current_lane - 1)
        self.update_lane_list()
        self.update_canvas()

    def on_canvas_click(self, event):
        if event.button() == Qt.LeftButton and self.current_lane < len(self.lane_points):
            x = int(event.pos().x())
            y = int(event.pos().y())
            self.push_undo()
            self.lane_points[self.current_lane].append((x, y))
            # sort points by y
            self.lane_points[self.current_lane].sort(key=lambda x: x[1])
            self.update_lane_list()  # 新增：及时更新车道线列表
            self.update_canvas()

    def update_canvas(self):
        if self.image is None:
            return
        img = self.image.copy()
        painter = QPainter()
        qimg = QImage(img.data, TUSIMPLE_IMG_SIZE[0], TUSIMPLE_IMG_SIZE[1], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        painter.begin(pixmap)

        # 只画水平参考线
        if hasattr(self, "h_samples") and self.h_samples:
            pen = QPen(QColor(200, 200, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            for i in range(len(self.h_samples)):
                if i % 2 == 0:
                    y = self.h_samples[i]
                    painter.drawLine(0, y, 1280, y)
                    painter.setPen(QColor(80, 80, 80))
                    painter.drawText(5, y - 2, f"{y}")
                    painter.setPen(pen)

        # 画车道线
        if self.select_all_checkbox is not None and self.select_all_checkbox.isChecked():
            lane_indices = range(len(self.lane_points))
        else:
            lane_indices = [self.current_lane] if 0 <= self.current_lane < len(self.lane_points) else []

        for idx in lane_indices:
            lane = self.lane_points[idx]
            color = LANE_COLORS[idx % len(LANE_COLORS)]
            pen = QPen(color, 3)
            painter.setPen(pen)
            for i in range(1, len(lane)):
                painter.drawLine(QPoint(*lane[i-1]), QPoint(*lane[i]))
            for pt in lane:
                painter.setBrush(color)
                painter.drawEllipse(QPoint(*pt), 2, 2)  # 修改：直径为3（半径为1）
        painter.end()
        self.canvas.setPixmap(pixmap)

    def push_undo(self):
        self.undo_stack.append(json.dumps(self.lane_points))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(json.dumps(self.lane_points))
        self.lane_points = json.loads(self.undo_stack.pop())
        self.update_lane_list()  # 新增：及时更新车道线列表
        self.update_canvas()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(json.dumps(self.lane_points))
        self.lane_points = json.loads(self.redo_stack.pop())
        self.update_lane_list()  # 新增：及时更新车道线列表
        self.update_canvas()

    def on_select_all_changed(self, state):
        if state == Qt.Checked:
            self.lane_list.setEnabled(False)
        else:
            self.lane_list.setEnabled(True)
        self.update_canvas()

    def show_current_lane_points(self):
        if 0 <= self.current_lane < len(self.lane_points):
            points = self.lane_points[self.current_lane]
            if not points:
                msg = self.lang_manager.get_text("msg_no_points")
            else:
                msg = "\n".join([f"({x}, {y})" for x, y in points])
        else:
            msg = self.lang_manager.get_text("msg_no_lane_selected")
        QMessageBox.information(self, 
            self.lang_manager.get_text("dialog_info"), msg)

    def organize_current_lane(self):
        """
        对当前选中车道线的像素点进行线性插值，生成tusimple特征点（h_samples对应的x），
        并用插值结果替换原有像素点列表。
        """
        if not (0 <= self.current_lane < len(self.lane_points)):
            QMessageBox.warning(self, 
                self.lang_manager.get_text("dialog_warning"),
                self.lang_manager.get_text("msg_no_lane_selected"))
            return
        if not self.h_samples or len(self.lane_points[self.current_lane]) < 2:
            QMessageBox.warning(self, 
                self.lang_manager.get_text("dialog_warning"),
                self.lang_manager.get_text("msg_insufficient_points"))
            return

        # 取出并排序当前车道线的点
        points = sorted(self.lane_points[self.current_lane], key=lambda x: x[1])
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]

        # 只对h_samples范围内插值
        min_y, max_y = min(ys), max(ys)
        interp_h_samples = [y for y in self.h_samples if min_y <= y <= max_y]
        if len(interp_h_samples) == 0:
            QMessageBox.warning(self, 
                self.lang_manager.get_text("dialog_warning"),
                self.lang_manager.get_text("msg_no_intersection"))
            return

        # 线性插值
        interp_xs = np.interp(interp_h_samples, ys, xs)
        new_points = [(int(round(x)), int(y)) for x, y in zip(interp_xs, interp_h_samples)]

        # 替换原有点
        self.push_undo()
        self.lane_points[self.current_lane] = new_points
        self.update_lane_list()  # 新增：及时更新车道线列表
        self.update_canvas()
        QMessageBox.information(self, 
            self.lang_manager.get_text("dialog_success"),
            self.lang_manager.get_text("msg_interpolation_success", 
                count=len(new_points)))

    def save_copy(self):
        copy_filename = self._save_copy()
        QMessageBox.information(self, "保存成功", f"副本已保存为：{copy_filename}")
        

    def _save_copy(self):
        """保存标注数据的副本，文件名为原文件名加上_tmp.json"""
        if not self.annotation_data:
            QMessageBox.warning(self, "警告", "没有标注数据可保存！")
            return

        # Check if current lanes exceed max_lanes
        if len(self.lane_points) > self.config["max_lanes"]:
            QMessageBox.warning(
                self,
                self.lang_manager.get_text("dialog_warning"),
                self.lang_manager.get_text("msg_too_many_lanes", 
                    current=len(self.lane_points), 
                    max=self.config["max_lanes"])
            )
            return

        # 获取当前json文件名
        current_json = self.json_file_path

        # 生成副本文件名
        project_id = self.config["project_id"]
        if current_json.endswith(f"{project_id}_tmp.json"):
            copy_filename = current_json
        else:
            copy_filename = f"{current_json}_{project_id}_tmp.json"
        copy_filepath = os.path.join(self.last_json_path, copy_filename)

        # 新增：保存前自动检查并插值
        self.auto_interpolate_all_lanes_to_h_samples()
        self.save_current_lane_points_to_annotation()

        # 保存副本
        with open(copy_filepath, "w") as f:
            for ann in self.annotation_data:
                json.dump(ann, f)
                f.write("\n")
        
        self.last_saved_lane_points = json.dumps(self.lane_points)  # 更新快照
        # 更新json_file_path
        self.json_file_path = copy_filepath
        self._open_annotation(copy_filepath)
        return copy_filename

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 
            self.lang_manager.get_text("dialog_exit"),
            self.lang_manager.get_text("msg_exit"),
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_cache()
            event.accept()
        else:
            event.ignore()

    def show_config_dialog(self):
        dialog = ConfigDialog(self.config, self)
        
        if dialog.exec_() == QDialog.Accepted:
            old_lang = self.config.get("lang", "CN")
            new_config = dialog.get_config()
            self.config.update(new_config)
            self.save_config()
            
            if old_lang != new_config["lang"]:
                QMessageBox.information(self, "提示", 
                    self.lang_manager.get_text("msg_lang_changed"))
            
            # 如果当前车道线数超过新的最大值，删除多余的车道线
            max_lanes = self.config["max_lanes"]
            if len(self.lane_points) > max_lanes:
                self.lane_points = self.lane_points[:max_lanes]
                self.update_lane_list()
                self.update_canvas()
                QMessageBox.information(self, "提示", f"已将车道线数量限制为{max_lanes}条")

    def load_config(self):
        """加载配置文件"""
        config_file = "config.json"
        default_config = {
            "image_root": "datasets/TUSimple/tusimple",
            "project_id": "tusimple_lane",
            "max_lanes": 6,
            "lang": "CN"  # 新增默认语言设置
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return default_config
        else:
            # 如果配置文件不存在，创建默认配置文件
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=4)
            return default_config

    def save_config(self):
        """保存配置到文件"""
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LaneLabelTool()
    win.show()
    sys.exit(app.exec_())






