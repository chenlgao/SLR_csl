# UI.py
import sys
import os
import cv2
import numpy as np
import speech_recognition as sr
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QImage, QPixmap
import torch
from torchvision import transforms
from PIL import Image
from CNN_LSTM import FeatureExtractor, MultiModalCNNTransformerModel
import mediapipe as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpeechThread(QThread):
    result_signal = pyqtSignal(str)

    def run(self):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
                try:
                    text = r.recognize_google(audio, language='zh-CN')
                    self.result_signal.emit(f"[语音] {text}")
                except sr.UnknownValueError:
                    self.result_signal.emit("[语音] 无法识别语音")
                except sr.RequestError:
                    self.result_signal.emit("[语音] 服务不可用")
        except Exception as e:
            self.result_signal.emit("[语音] 麦克风访问失败")

class CameraThread(QThread):
    result_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(QImage)
    result_signal = pyqtSignal(str, float)  # 新增float参数

    def __init__(self, model, feature_extractor, label_map):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.label_map = label_map
        self.running = True
        self.frame_buffer = []
        self.keypoint_buffer = []
        self.max_sequence_length = 170
        self.predict_interval = 170
        self.frame_counter = 0
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_signal.emit("[错误] 无法打开摄像头")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.result_signal.emit("[错误] 视频流中断")
                    break

                # 显示实时画面
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_signal.emit(qt_image.scaled(640, 480, Qt.KeepAspectRatio))

                # 处理帧和关键点
                processed_frame, keypoints = self._process_frame(frame)
                self.frame_buffer.append(processed_frame)
                self.keypoint_buffer.append(keypoints)

                # 保持固定序列长度
                if len(self.frame_buffer) > self.max_sequence_length:
                    self.frame_buffer = self.frame_buffer[-self.max_sequence_length:]
                    self.keypoint_buffer = self.keypoint_buffer[-self.max_sequence_length:]
                
                # 定期预测
                self.frame_counter += 1
                if self.frame_counter % self.predict_interval == 0:
                    pred_label, confidence = self._predict()
                    self.result_signal.emit(pred_label, confidence)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mp_hands.close()

    def _process_frame(self, frame):
        """处理单帧并提取关键点"""
        # 预处理视觉帧
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        processed_tensor = transform(pil_image)

        # 提取关键点
        hands_results = self.mp_hands.process(frame_rgb)
        keypoints = np.zeros(126, dtype=np.float32)
        if hands_results.multi_hand_landmarks:
            hands = hands_results.multi_hand_landmarks[:2]
            for hand_idx, hand in enumerate(hands):
                start = hand_idx * 63
                for lm_idx, landmark in enumerate(hand.landmark[:21]):
                    pos = start + lm_idx * 3
                    if pos + 2 < 126:
                        keypoints[pos] = landmark.x
                        keypoints[pos+1] = landmark.y
                        keypoints[pos+2] = landmark.z

        return processed_tensor, torch.tensor(keypoints, dtype=torch.float32)

    def _predict(self):
        """执行多模态预测"""
        if len(self.frame_buffer) < 10:
            return "等待更多帧..."

        # 转换为Tensor
        frames_tensor = torch.stack(self.frame_buffer).to(device)  # [seq_len, C, H, W]
        keypoints_tensor = torch.stack(self.keypoint_buffer).to(device)  # [seq_len, 126]

        with torch.no_grad():
            # 提取视觉特征
            visual_features = self.feature_extractor(
                frames_tensor.view(-1, 3, 256, 256)
            ).view(1, -1, 512)  # [1, seq_len, 512]

            # 拼接关键点特征
            combined_features = torch.cat([
                visual_features,
                keypoints_tensor.unsqueeze(0)
            ], dim=2)  # [1, seq_len, 638]

            # 调整序列长度
            seq_len = combined_features.shape[1]
            if seq_len < self.max_sequence_length:
                padding = torch.zeros(1, self.max_sequence_length-seq_len, 638).to(device)
                combined_features = torch.cat([combined_features, padding], dim=1)
            else:
                combined_features = combined_features[:, :self.max_sequence_length]

            outputs = self.model(combined_features)
            probs = torch.softmax(outputs, dim=1)
            

        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item() 
        return self.label_map.get(pred_idx, "未知标签"), confidence

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.model = None
        self.preview_label = QLabel()
        self.camera_thread = None
        self.label_map = {}
        self.initUI()
        self.load_models()
    
    def append_history_with_confidence(self, label, confidence):
        """添加带置信度的历史记录"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = f"[{timestamp}] [实时] {label} (置信度: {confidence:.2%})"
        
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text + "\n")
        self.history.ensureCursorVisible()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("手语识别系统 v3.0")
        self.setGeometry(100, 100, 1024, 768)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QVBoxLayout()
        control_panel.addWidget(QLabel("操作选项：", styleSheet="font-weight: bold;"))
        
        # 功能按钮
        self.btn_video = QPushButton("📁 导入视频文件")
        self.btn_image = QPushButton("📷 导入图片序列")
        self.btn_camera = QPushButton("🎥 开启实时识别")
        self.btn_voice = QPushButton("🎤 语音输入")
        
        # 按钮样式
        button_style = """
            QPushButton { 
                padding: 10px; 
                font-size: 14px; 
                min-width: 150px;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover { background-color: #f0f0f0; }
        """
        for btn in [self.btn_video, self.btn_image, self.btn_camera, self.btn_voice]:
            btn.setStyleSheet(button_style)
        
        # 绑定事件
        self.btn_video.clicked.connect(self.select_video)
        self.btn_image.clicked.connect(self.select_images)
        self.btn_camera.clicked.connect(self.toggle_camera)
        self.btn_voice.clicked.connect(self.speech_input)

        # 信息提示
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        
        # 历史记录
        self.history = QTextEdit()
        self.history.setStyleSheet("""
            font-family: 'Microsoft YaHei'; 
            font-size: 12px; 
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 5px;
        """)
        self.history.setPlaceholderText("识别日志将显示在此处...")

        # 右侧预览面板
        preview_panel = QVBoxLayout()
        preview_panel.addWidget(QLabel("实时预览：", styleSheet="font-weight: bold;"))
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("""
            background: #333; 
            border-radius: 10px;
            border: 2px solid #666;
        """)
        preview_panel.addWidget(self.preview_label)

        # 组装界面
        control_panel.addWidget(self.btn_video)
        control_panel.addWidget(self.btn_image)
        control_panel.addWidget(self.btn_camera)
        control_panel.addWidget(self.btn_voice)
        control_panel.addWidget(self.lbl_status)
        control_panel.addWidget(QLabel("操作记录：", styleSheet="font-weight: bold;"))
        control_panel.addWidget(self.history)

        main_layout.addLayout(control_panel, 35)
        main_layout.addLayout(preview_panel, 65)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.btn_test = QPushButton("🔧 运行诊断测试")
        self.btn_test.setStyleSheet("background-color: #ff9900;")
        self.btn_test.clicked.connect(self.run_diagnostic_test)
        control_panel.addWidget(self.btn_test)

    def load_models(self):
        """加载预训练模型"""
        try:
            model_path = r'D:\ctcn_2\models\best_model.pth'
            checkpoint = torch.load(model_path, map_location=device)
            
            # 初始化特征提取器
            self.feature_extractor = FeatureExtractor(model_name='resnet18').to(device)
            self.feature_extractor.eval()
            
            # 初始化Transformer模型
            self.model = MultiModalCNNTransformerModel(
                feature_dim=638,
                num_classes=len(checkpoint['index_to_label']),
                heads=2
            ).to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 加载标签映射
            self.label_map = {int(k): v for k, v in checkpoint['index_to_label'].items()}

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            sys.exit(1)

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.quit()
            self.camera_thread.wait()
            self.btn_camera.setText("🎥 开启实时识别")
            self.lbl_status.setText("已停止摄像头识别")
        else:
            self.start_camera()
            self.btn_camera.setText("⏹️ 停止识别")
            self.lbl_status.setText("实时识别运行中...")

    def start_camera(self):
        """启动摄像头线程"""
        self.camera_thread = CameraThread(
            model=self.model,
            feature_extractor=self.feature_extractor,
            label_map=self.label_map
        )
        # 绑定新信号（带置信度）
        self.camera_thread.frame_signal.connect(self.update_preview)
        self.camera_thread.result_signal.connect(self.append_history_with_confidence)  # 正确位置
        self.camera_thread.start()

    def select_video(self):
        """选择视频文件"""
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "选择视频文件", 
                "", 
                "视频文件 (*.mp4 *.avi);;所有文件 (*)"
            )
            if path:
                if not os.path.exists(path):
                    raise FileNotFoundError("文件不存在")

                # 执行预测
                result = self._predict_media(path, is_video=True)
                self.append_history(f"[视频识别] 结果: {result}")

        except Exception as e:
            self.show_error("视频处理错误", str(e))

    def select_images(self):
        """选择图片序列"""
        try:
            path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
            if path:
                result = self._predict_media(path, is_video=False)
                self.append_history(f"[图片识别] 结果: {result}")

        except Exception as e:
            self.show_error("图片处理错误", str(e))

    def _predict_media(self, input_path, is_video):
        """统一媒体预测方法"""
        mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
    
        frames = []
        keypoints_list = []
        
        try:
            if is_video:
                cap = cv2.VideoCapture(input_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame, keypoints = self._process_frame_with_keypoints(frame, mp_hands)
                    frames.append(processed_frame)
                    keypoints_list.append(keypoints)
                cap.release()
            else:
                valid_exts = ('.png', '.jpg', '.jpeg')
                for fname in sorted(os.listdir(input_path)):
                    if fname.lower().endswith(valid_exts):
                        frame = cv2.imread(os.path.join(input_path, fname))
                        if frame is not None:
                            processed_frame, keypoints = self._process_frame_with_keypoints(frame, mp_hands)
                            frames.append(processed_frame)
                            keypoints_list.append(keypoints)
            
            if not frames:
                return "未检测到有效输入"

            # 转换为Tensor并处理
            frames_tensor = torch.stack(frames).to(device)
            keypoints_tensor = torch.stack(keypoints_list).to(device)

            with torch.no_grad():
                # 提取并拼接特征
                visual_features = self.feature_extractor(
                    frames_tensor.view(-1, 3, 256, 256)
                ).view(1, -1, 512)
                combined_features = torch.cat([
                    visual_features,
                    keypoints_tensor.unsqueeze(0)
                ], dim=2)

                # 调整序列长度
                seq_len = combined_features.shape[1]
                if seq_len < 170:
                    padding = torch.zeros(1, 170-seq_len, 638).to(device)
                    combined_features = torch.cat([combined_features, padding], dim=1)
                else:
                    combined_features = combined_features[:, :170]

                outputs = self.model(combined_features)
                probs = torch.softmax(outputs, dim=1)
            
            # 处理预测结果
            pred_idx = torch.argmax(probs).item()
            label = self.label_map.get(pred_idx, "未知标签")
            confidence = probs[0][pred_idx].item()
            return f"识别结果: {label} (置信度: {confidence:.2%})"

        except Exception as e:
            return f"错误: {str(e)}"
        finally:
            mp_hands.close()
    def _preprocess_frame(self, frame):
        """预处理单帧图像"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return transform(pil_image)
    def _process_frame_with_keypoints(self, frame, mp_hands):
        """提取单帧+关键点"""
        # 预处理视觉帧
        processed_tensor = self._preprocess_frame(frame)
        
        # 提取关键点
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = mp_hands.process(frame_rgb)
        
        # 初始化关键点数组（126维）
        keypoints = np.zeros(126, dtype=np.float32)
        
        if hands_results.multi_hand_landmarks:
            hands = hands_results.multi_hand_landmarks[:2]
            for hand_idx, hand in enumerate(hands):
                start = hand_idx * 63
                for lm_idx, landmark in enumerate(hand.landmark[:21]):
                    pos = start + lm_idx * 3
                    if pos + 2 < 126:
                        keypoints[pos] = landmark.x
                        keypoints[pos+1] = landmark.y
                        keypoints[pos+2] = landmark.z
        
        return processed_tensor, torch.tensor(keypoints, dtype=torch.float32)

    def speech_input(self):
        """语音输入处理"""
        try:
            self.btn_voice.setEnabled(False)
            self.btn_voice.setText("正在监听...")
            self.lbl_status.setText("请开始说话（最长8秒）")
            self.append_history("[语音] 正在识别中...")
            
            self.speech_thread = SpeechThread()
            self.speech_thread.result_signal.connect(self.handle_speech_result)
            self.speech_thread.start()
            
        except Exception as e:
            self.show_error("语音输入错误", str(e))
            self.reset_voice_button()

    def run_diagnostic_test(self):
        """诊断测试"""
        # 生成随机测试数据
        dummy_input = torch.randn(1, 170, 638).to(device)  # 注意维度改为638
        with torch.no_grad():
            outputs = self.model(dummy_input)
            probs = torch.softmax(outputs, dim=1)
        
        print("\n诊断测试结果:")
        print("Logits:", outputs)
        print("概率分布:", probs)

    def handle_speech_result(self, text):
        """处理语音识别结果"""
        self.append_history(text)
        self.reset_voice_button()

    def reset_voice_button(self):
        """重置语音按钮状态"""
        self.btn_voice.setEnabled(True)
        self.btn_voice.setText("🎤 语音输入")
        self.lbl_status.setText("就绪")

    def show_error(self, title, message):
        """显示错误弹窗"""
        QMessageBox.critical(self, title, message)
        self.append_history(f"[错误] {title}: {message}")

    def update_preview(self, image):
        """更新预览画面"""
        self.preview_label.setPixmap(QPixmap.fromImage(image))

    def append_history(self, text):
        """添加历史记录"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"[{timestamp}] {text}\n")
        self.history.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())