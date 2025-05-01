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
    QVBoxLayout, QHBoxLayout, QListWidget, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint

LANE_COLORS = [
    QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
    QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)
]

TUSIMPLE_IMG_SIZE = (1280, 720)
CANVAS_SIZE = (960, 540)

class LaneLabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车道线标注工具 Lane Annotation Editor")
        #self.resize(1200, 800)
        #self.setMinimumSize(1600, 900)
        self.resize(1600, 900)
        self.annotation_data = []
        self.current_index = 0
        self.current_lane = 0
        self.undo_stack = []
        self.redo_stack = []
        self.image = None
        self.img_h_scale = CANVAS_SIZE[1] / TUSIMPLE_IMG_SIZE[1]
        self.img_w_scale = CANVAS_SIZE[0] / TUSIMPLE_IMG_SIZE[0]
        self.image_path = ""
        self.h_samples = []
        self.lane_points = []  # [[(x1, y1), (x2, y2), ...], ...]
        self.path_label = QLabel("")  # 新增：用于显示路径和分辨率
        self.json_file_label = QLabel("")  # 新增：用于显示json文件名
        self.json_file_label.setAlignment(Qt.AlignLeft)
        self.last_json_path = self.load_last_json_path()  # 初始化时从cache.json加载
        self.init_ui()

    def init_ui(self):
        # 顶部按钮
        open_btn = QPushButton("打开标注文件")
        open_btn.clicked.connect(self.open_annotation)
        save_btn = QPushButton("保存标注")
        save_btn.clicked.connect(self.save_annotation)
        prev_btn = QPushButton("上一张")
        prev_btn.clicked.connect(self.prev_image)
        next_btn = QPushButton("下一张")
        next_btn.clicked.connect(self.next_image)
        self.image_label = QLabel("未加载图片")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.path_label.setAlignment(Qt.AlignLeft)  # 新增：左对齐

        # 右侧面板
        self.lane_list = QListWidget()
        self.lane_list.currentRowChanged.connect(self.select_lane)
        add_lane_btn = QPushButton("添加车道线")
        add_lane_btn.clicked.connect(self.add_lane)
        del_lane_btn = QPushButton("删除车道线")
        del_lane_btn.clicked.connect(self.delete_lane)
        undo_btn = QPushButton("撤销")
        undo_btn.clicked.connect(self.undo)
        redo_btn = QPushButton("重做")
        redo_btn.clicked.connect(self.redo)

        # 布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(open_btn)
        top_layout.addWidget(save_btn)
        top_layout.addWidget(prev_btn)
        top_layout.addWidget(next_btn)

        # 新增：按钮下方显示json文件名、路径和分辨率
        path_layout = QVBoxLayout()
        path_layout.addLayout(top_layout)
        path_layout.addWidget(self.json_file_label)  # 新增：添加json文件名label
        path_layout.addWidget(self.path_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("车道线列表"))
        right_layout.addWidget(self.lane_list)
        right_layout.addWidget(add_lane_btn)
        right_layout.addWidget(del_lane_btn)
        right_layout.addWidget(undo_btn)
        right_layout.addWidget(redo_btn)
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

    def load_last_json_path(self):
        cache_file = "cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
                    return cache.get("last_json_path", "")
            except Exception:
                return ""
        return ""

    def save_last_json_path(self, path):
        cache = {"last_json_path": path}
        with open("cache.json", "w") as f:
            json.dump(cache, f)

    def open_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择TuSimple标注文件", self.last_json_path, "JSON Files (*.json)"
        )
        if not file_path:
            return
        self.last_json_path = os.path.dirname(file_path)
        self.save_last_json_path(self.last_json_path)  # 保存到cache.json
        self.annotation_data = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.annotation_data.append(json.loads(line))
        self.current_index = 0
        # 新增：显示json文件名
        self.json_file_label.setText(f"JSON文件: {os.path.basename(file_path)}")
        self.load_image_and_lanes()

    def save_annotation(self):
        if not self.annotation_data:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存标注文件", "", "JSON Files (*.json)")
        if not file_path:
            return
        self.last_json_path = os.path.dirname(file_path)
        self.save_last_json_path(self.last_json_path)  # 保存到cache.json
        with open(file_path, "w") as f:
            json.dump(self.annotation_data, f, indent=2)
        QMessageBox.information(self, "保存成功", "标注已保存！")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image_and_lanes()

    def next_image(self):
        if self.current_index < len(self.annotation_data) - 1:
            self.current_index += 1
            self.load_image_and_lanes()

    def load_image_and_lanes(self):
        ann = self.annotation_data[self.current_index]
        self.image_path = os.path.join("datasets/TUSimple/tusimple", ann["raw_file"])
        self.h_samples = ann["h_samples"]
        self.lane_points = []
        for lane in ann["lanes"]:
            points = []
            for x, y in zip(lane, self.h_samples):
                if x >= 0:
                    points.append((x, y))
            self.lane_points.append(points)
        self.current_lane = 0
        self.update_lane_list()
        self.load_image()
        self.update_canvas()
        # 更新路径和分辨率显示
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.path_label.setText(f"Image: {self.image_path}    {w}x{h}")
        else:
            self.path_label.setText(f"Image: {self.image_path}    (未加载)")

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
            self.lane_list.addItem(f"车道线 {idx+1} ({len(lane)}点)")
        self.lane_list.setCurrentRow(self.current_lane)

    def select_lane(self, idx):
        if 0 <= idx < len(self.lane_points):
            self.current_lane = idx
            self.update_canvas()

    def add_lane(self):
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
            self.update_canvas()

    def update_canvas(self):
        if self.image is None:
            return
        img = self.image.copy()
        painter = QPainter()    
        #qimg = QImage(img.data, CANVAS_SIZE[0], CANVAS_SIZE[1], img.strides[0], QImage.Format_RGB888)
        qimg = QImage(img.data, TUSIMPLE_IMG_SIZE[0], TUSIMPLE_IMG_SIZE[1], img.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        painter.begin(pixmap)

        # 只画水平参考线
        if hasattr(self, "h_samples") and self.h_samples:
            pen = QPen(QColor(200, 200, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            #for y in self.h_samples:
            for i in range(len(self.h_samples)):
                if i % 2 == 0:
                    y = self.h_samples[i]
                    painter.drawLine(0, y, 1280, y)                    
                    # 新增：在左侧显示y值
                    painter.setPen(QColor(80, 80, 80))
                    painter.drawText(5, y - 2, f"{y}")
                    painter.setPen(pen)  # 恢复参考线颜色

        # 画车道线
        for idx, lane in enumerate(self.lane_points):
            color = LANE_COLORS[idx % len(LANE_COLORS)]
            pen = QPen(color, 3)
            painter.setPen(pen)
            for i in range(1, len(lane)):
                painter.drawLine(QPoint(*lane[i-1]), QPoint(*lane[i]))
            for pt in lane:
                painter.setBrush(color)
                painter.drawEllipse(QPoint(*pt), 5, 5)
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
        self.update_lane_list()
        self.update_canvas()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(json.dumps(self.lane_points))
        self.lane_points = json.loads(self.redo_stack.pop())
        self.update_lane_list()
        self.update_canvas()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出', '确定要退出吗？未保存的更改将丢失。',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LaneLabelTool()
    win.show()
    sys.exit(app.exec_())






