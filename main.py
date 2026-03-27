import sys, json, os
import queue
import numpy as np
import pyaudiowpatch as pyaudio
from scipy import signal as scipy_signal
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QComboBox, QTextEdit, QLabel, QPushButton,
                             QHBoxLayout, QSlider, QScrollArea)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint, QTimer, QSize, QRectF
from PyQt6.QtGui import QTextOption, QPainter, QPainterPath, QPen, QFontMetrics, QFontMetricsF, QFont, QColor

# ---- Settings Management ----
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_SETTINGS = {
    "font_size": 36,
    "line_spacing": 1.5,
    "segment_spacing": 25,
    "bg_alpha": 160,
    "outline_width": 5,
    "window_size": [800, 300],
    "window_pos": [100, 100]
}

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
    except Exception as e:
        print(f"Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

TARGET_SR = 16000

# ---------------------------------------------------------
# Worker Thread: Audio Capture (WASAPI Loopback)
# ---------------------------------------------------------
class AudioWorker(QThread):
    def __init__(self, device_index, device_sr, audio_queue):
        super().__init__()
        self.device_index = device_index
        self.device_sr = int(device_sr)
        self.audio_queue = audio_queue
        self.is_running = True

    def run(self):
        pa = pyaudio.PyAudio()
        try:
            device_info = pa.get_device_info_by_index(self.device_index)
            channels = int(device_info.get("maxInputChannels", 2))
            if channels == 0:
                channels = 2
            chunk_size = int(self.device_sr * 0.5)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=self.device_sr,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=chunk_size,
            )

            while self.is_running:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                if self.device_sr != TARGET_SR:
                    num_samples = int(len(audio) * TARGET_SR / self.device_sr)
                    audio = scipy_signal.resample(audio, num_samples)
                self.audio_queue.put(audio.astype(np.float32))

            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Ошибка захвата звука: {e}")
        finally:
            pa.terminate()

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()


# ---------------------------------------------------------
# Worker Thread: Whisper Recognition
# ---------------------------------------------------------
class WhisperWorker(QThread):
    text_recognized = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.is_running = True
        self.model = None

    def run(self):
        self.status_updated.emit("Загрузка модели Whisper (первый раз — несколько минут)...")

        self.model = None
        for device, compute, label in [("cuda", "float16", "GPU"), ("cpu", "int8", "CPU")]:
            try:
                candidate = WhisperModel("base", device=device, compute_type=compute)
                dummy = np.zeros(TARGET_SR, dtype=np.float32)
                list(candidate.transcribe(dummy, vad_filter=True)[0])
                self.model = candidate
                self.status_updated.emit(f"Модель загружена ({label}). Слушаю...")
                break
            except Exception:
                if device == "cuda":
                    self.status_updated.emit("GPU недоступен, переключаюсь на CPU...")
                else:
                    self.status_updated.emit("Ошибка инициализации модели.")
                    return

        if self.model is None:
            return

        buffer = []
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                buffer.append(chunk)
                if len(buffer) >= 6:
                    audio_segment = np.concatenate(buffer)
                    segments, _ = self.model.transcribe(
                        audio_segment,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False,
                        no_repeat_ngram_size=3,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,
                    )
                    text = " ".join([s.text for s in segments])
                    if text.strip():
                        self.text_recognized.emit(text.strip())
                    buffer = buffer[-2:]
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при распознавании: {e}")

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()


# ---------------------------------------------------------
# Worker Thread: Translation to Hebrew
# ---------------------------------------------------------
class TranslationWorker(QThread):
    translation_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.translation_queue = queue.Queue()
        self.is_running = True
        self.translator = GoogleTranslator(source="auto", target="iw")

    def translate(self, text):
        self.translation_queue.put(text)

    def run(self):
        while self.is_running:
            try:
                text = self.translation_queue.get(timeout=0.5)
                translated = self.translator.translate(text)
                if translated and translated.strip():
                    self.translation_ready.emit(translated.strip())
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка перевода: {e}")

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()


# ---------------------------------------------------------
# Custom Label with Text Outline (Stroke)
# ---------------------------------------------------------
class StrokeLabel(QLabel):
    def __init__(self, text="", font_size=32, line_spacing=1.5, outline_width=5, parent=None):
        super().__init__(text, parent)
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.outline_width = outline_width
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: transparent; color: white;")
        self.setWordWrap(False) # Manual wrapping only

    def set_font_size(self, size):
        self.font_size = size
        self.update_all()

    def set_line_spacing(self, spacing):
        self.line_spacing = spacing
        self.update_all()

    def set_outline_width(self, width):
        self.outline_width = width
        self.update_all()

    def update_all(self):
        self.update()
        self.updateGeometry()

    def sizeHint(self):
        font = QFont(self.font())
        font.setPointSize(self.font_size)
        font.setBold(True)
        metrics = QFontMetricsF(font)
        
        # Robust line height: lineSpacing + stroke + 15% safety gap
        line_h = metrics.lineSpacing() + self.outline_width + (self.font_size * 0.15)
        
        # Approximate size with wrapping
        max_w = self.width() if self.width() > 100 else 800
        words = self.text().split(' ')
        lines_count = 1
        curr_w = 0
        for w in words:
            w_w = metrics.horizontalAdvance(w + " ") + self.outline_width
            if curr_w + w_w > max_w - 40:
                lines_count += 1
                curr_w = w_w
            else:
                curr_w += w_w
        
        total_h = int(lines_count * line_h) + self.outline_width + 40
        return QSize(max_w, total_h)

    def paintEvent(self, event):
        if not self.text():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont(self.font())
        font.setPointSize(self.font_size)
        font.setBold(True)
        
        metrics = QFontMetricsF(font)
        line_h = metrics.lineSpacing() + self.outline_width + (self.font_size * 0.15)

        # Handle word wrapping manually for the painter path
        words = self.text().split(' ')
        lines = []
        current_line = ""
        max_width = self.width() - 40 - self.outline_width
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            if metrics.horizontalAdvance(test_line) < max_width:
                current_line = test_line
            else:
                if current_line: lines.append(current_line)
                current_line = word
        if current_line: lines.append(current_line)

        total_height = len(lines) * line_h
        # Adjust start_y: center the block of text within the label
        start_y = (self.height() - total_height) / 2 + metrics.ascent() + (self.font_size * 0.07)

        for i, line in enumerate(lines):
            path = QPainterPath()
            line_w = metrics.horizontalAdvance(line)
            # Center each line horizontally
            x = (self.width() - line_w) / 2
            y = start_y + i * line_h
            path.addText(x, y, font, line)

            # Customizable outline
            pen = QPen(QColor(0, 0, 0), self.outline_width)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

            # White fill
            painter.fillPath(path, QColor(255, 255, 255))


# ---------------------------------------------------------
# Hebrew Translation Window (subtitle-style, transparent)
# ---------------------------------------------------------
class HebrewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("תרגום לעברית")
        
        # Load persistent settings
        self.settings = load_settings()
        self._font_size = self.settings["font_size"]
        self._line_spacing = self.settings["line_spacing"]
        self._segment_spacing = self.settings["segment_spacing"]
        self._bg_alpha = self.settings["bg_alpha"]
        self._outline_width = self.settings["outline_width"]

        self.resize(self.settings["window_size"][0], self.settings["window_size"][1])
        self.move(self.settings["window_pos"][0], self.settings["window_pos"][1])
        
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.NoDropShadowWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        # --- UI Setup ---
        central = QWidget()
        central.setObjectName("subtitle_bg")
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 10, 15, 15)
        layout.setSpacing(0)

        # Control Bar (Hidden by default, shown on hover)
        self.ctrl_widget = QWidget()
        self.ctrl_widget.setFixedHeight(40)
        self.ctrl_widget.setVisible(False)
        ctrl_layout = QHBoxLayout(self.ctrl_widget)
        ctrl_layout.setContentsMargins(10, 0, 10, 0)
        
        # Drag Handle / Label
        drag_lbl = QLabel("⠿ תרגום")
        drag_lbl.setStyleSheet("color: rgba(255,255,255,150); font-size: 11px; font-weight: bold;")
        ctrl_layout.addWidget(drag_lbl)
        
        ctrl_layout.addStretch()
        
        # Settings Gear Button
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setToolTip("הגדרות")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setCheckable(True)
        self.settings_btn.setStyleSheet("""
            QPushButton { 
                background: transparent; color: white; font-size: 20px; border-radius: 15px;
            }
            QPushButton:hover { background: rgba(255,255,255,50); }
            QPushButton:checked { background: rgba(255,255,255,80); color: #00ffcc; }
        """)
        self.settings_btn.clicked.connect(self._toggle_settings)
        ctrl_layout.addWidget(self.settings_btn)

        layout.addWidget(self.ctrl_widget)

        # ---- SETTINGS PANEL ----
        self.settings_panel = QWidget()
        self.settings_panel.setVisible(False)
        self.settings_panel.setStyleSheet("""
            QWidget { background: rgba(30, 30, 30, 230); border-radius: 10px; color: white; }
            QLabel { background: transparent; font-size: 11px; color: #aaa; }
            QSlider::handle:horizontal { background: #00ffcc; border-radius: 5px; }
        """)
        s_layout = QVBoxLayout(self.settings_panel)
        s_layout.setContentsMargins(15, 15, 15, 15)
        s_layout.setSpacing(8)

        def add_setting(label_text, min_v, max_v, curr_v, callback, factor=1):
            lbl = QLabel(label_text)
            s_layout.addWidget(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(int(curr_v * factor))
            slider.valueChanged.connect(lambda v: callback(v / factor))
            s_layout.addWidget(slider)
            return slider

        self.slider_font = add_setting("גודל גופן", 12, 120, self._font_size, self._update_font_size)
        self.slider_line = add_setting("מרווח בין שורות", 10, 30, self._line_spacing, self._update_line_spacing, 10)
        self.slider_segment = add_setting("מרווח בין פסקאות", 0, 100, self._segment_spacing, self._update_segment_spacing)
        self.slider_outline = add_setting("עובי קו מתאר", 0, 15, self._outline_width, self._update_outline_width)
        self.slider_bg = add_setting("שקיפות רקע", 0, 255, self._bg_alpha, self._update_bg)

        layout.addWidget(self.settings_panel)

        # ---- Scrollable area for history ----
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Scroll bar styling
        self.scroll_area.verticalScrollBar().setStyleSheet(
            "QScrollBar:vertical { width: 6px; background: rgba(255,255,255,20); border-radius: 3px; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,100); border-radius: 3px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )

        self.history_widget = QWidget()
        self.history_widget.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(self._segment_spacing)
        self.history_layout.addStretch() # Push everything to bottom
        
        self.scroll_area.setWidget(self.history_widget)
        layout.addWidget(self.scroll_area, stretch=1)

        self._update_bg(self._bg_alpha)
        
        # Auto-hide timer
        self.hide_timer = QTimer()
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._maybe_hide_controls)
        
        self.setMouseTracking(True)
        self.setToolTip("גרור להזזה") # Drag to move
        central.setMouseTracking(True)
        
        self._resizing = False
        self._drag_pos = None

    def clear_history(self):
        # Remove all widgets except the last stretch
        while self.history_layout.count() > 1:
            item = self.history_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._scroll_to_bottom()

    def _update_bg(self, value):
        self._bg_alpha = int(value)
        self.settings["bg_alpha"] = self._bg_alpha
        self.centralWidget().setStyleSheet(
            f"QWidget#subtitle_bg {{"
            f"  background-color: rgba(0, 0, 0, {self._bg_alpha});"
            f"  border-radius: 14px;"
            f"}}"
        )
        save_settings(self.settings)

    def _update_font_size(self, value):
        self._font_size = int(value)
        self.settings["font_size"] = self._font_size
        # Update all existing labels
        for i in range(self.history_layout.count()):
            item = self.history_layout.itemAt(i)
            if item and item.widget():
                if isinstance(item.widget(), StrokeLabel):
                    item.widget().set_font_size(self._font_size)
        save_settings(self.settings)

    def _update_line_spacing(self, value):
        self._line_spacing = value
        self.settings["line_spacing"] = self._line_spacing
        for i in range(self.history_layout.count()):
            item = self.history_layout.itemAt(i)
            if item and item.widget():
                if isinstance(item.widget(), StrokeLabel):
                    item.widget().set_line_spacing(self._line_spacing)
        save_settings(self.settings)

    def _update_segment_spacing(self, value):
        self._segment_spacing = int(value)
        self.settings["segment_spacing"] = self._segment_spacing
        self.history_layout.setSpacing(self._segment_spacing)
        save_settings(self.settings)

    def _update_outline_width(self, value):
        self._outline_width = int(value)
        self.settings["outline_width"] = self._outline_width
        for i in range(self.history_layout.count()):
            item = self.history_layout.itemAt(i)
            if item and item.widget():
                if isinstance(item.widget(), StrokeLabel):
                    item.widget().set_outline_width(self._outline_width)
        save_settings(self.settings)

    def _toggle_settings(self, checked):
        self.settings_panel.setVisible(checked)
        if checked:
            self.hide_timer.stop()
        else:
            self.hide_timer.start(3000)

    def _show_controls(self):
        self.ctrl_widget.setVisible(True)
        self.hide_timer.stop()

    def _maybe_hide_controls(self):
        # Don't hide if settings panel is open
        if self.settings_btn.isChecked():
            return
        self.ctrl_widget.setVisible(False)
        self.settings_panel.setVisible(False)

    def _hide_controls(self):
        # Legacy placeholder
        self._maybe_hide_controls()

    def _reset_hide_timer(self):
        if self.ctrl_widget.isVisible():
            self.hide_timer.start(3000)

    def enterEvent(self, event):
        self._show_controls()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # We don't hide immediately on leave, let the timer handle it
        # or hide faster if we want. Let's keep the timer.
        self.hide_timer.start(500) # Shorter timer on leave
        super().leaveEvent(event)

    # ---- Drag & Resize support ----
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            rect = self.rect()
            edge_margin = 15
            
            # Check if we're clicking near the bottom-right corner for resizing
            if (rect.width() - pos.x() < edge_margin and 
                rect.height() - pos.y() < edge_margin):
                self._resizing = True
            else:
                self._resizing = False
            self._drag_pos = event.globalPosition().toPoint()
            self._show_controls()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            if self._resizing:
                diff = event.globalPosition().toPoint() - self._drag_pos
                new_w = max(400, self.width() + diff.x())
                new_h = max(80, self.height() + diff.y()) # Lowered min height for slim look
                self.resize(new_w, new_h)
                self._drag_pos = event.globalPosition().toPoint()
                self.settings["window_size"] = [new_w, new_h]
                save_settings(self.settings)
            else:
                self.move(self.pos() + (event.globalPosition().toPoint() - self._drag_pos))
                self._drag_pos = event.globalPosition().toPoint()
                self.settings["window_pos"] = [self.x(), self.y()]
                save_settings(self.settings)
        
        # Check for resize area (bottom-right 20x20)
        if event.pos().x() > self.width() - 20 and event.pos().y() > self.height() - 20:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        self._show_controls()
        self.hide_timer.start(3000)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        self._resizing = False

    def append_text(self, text):
        if not text.strip():
            return
            
        # Create a new label for history
        lbl = StrokeLabel(text, 
                         font_size=self._font_size, 
                         line_spacing=self._line_spacing,
                         outline_width=self._outline_width)
        # Add before the last stretch
        count = self.history_layout.count()
        self.history_layout.insertWidget(count - 1, lbl)
        
        # Limit history to 2 segments (one top, one bottom)
        # count includes labels + 1 stretch
        if self.history_layout.count() > 3: # 2 labels + 1 stretch
            item = self.history_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        
        # Auto-scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        v_bar = self.scroll_area.verticalScrollBar()
        v_bar.setValue(v_bar.maximum())


# ---------------------------------------------------------
# Main GUI Window
# ---------------------------------------------------------
class LiveTranslateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Translate")
        self.resize(650, 350)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Устройство:"))
        self.device_combo = QComboBox()
        self.populate_devices()
        device_layout.addWidget(self.device_combo, stretch=1)
        layout.addLayout(device_layout)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Старт")
        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.lbl_status = QLabel("Ожидание...")
        self.lbl_status.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_status)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        font = self.text_edit.font()
        font.setPointSize(12)
        self.text_edit.setFont(font)
        layout.addWidget(self.text_edit)

        self.audio_queue = queue.Queue()
        self.audio_worker = None
        self.whisper_worker = None
        self.translation_worker = None

        # Открываем окно перевода сразу
        self.hebrew_window = HebrewWindow()
        self.hebrew_window.show()

    def populate_devices(self):
        pa = pyaudio.PyAudio()
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            self.device_combo.addItem("WASAPI не найден", (-1, 48000))
            pa.terminate()
            return

        for i in range(pa.get_device_count()):
            device = pa.get_device_info_by_index(i)
            if (device.get("hostApi") == wasapi_info["index"]
                    and device.get("isLoopbackDevice", False)):
                sr = int(device.get("defaultSampleRate", 48000))
                self.device_combo.addItem(device["name"], (i, sr))

        pa.terminate()

        if self.device_combo.count() == 0:
            self.device_combo.addItem("Нет loopback-устройств", (-1, 48000))

    def start_capture(self):
        device_data = self.device_combo.currentData()
        if not device_data or device_data[0] == -1:
            return

        device_index, device_sr = device_data
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.text_edit.clear()
        self.hebrew_window.clear_history()

        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # Поток перевода
        self.translation_worker = TranslationWorker()
        self.translation_worker.translation_ready.connect(self.hebrew_window.append_text)
        self.translation_worker.start()

        # Поток Whisper
        self.whisper_worker = WhisperWorker(self.audio_queue)
        self.whisper_worker.text_recognized.connect(self.append_text)
        self.whisper_worker.text_recognized.connect(self.translation_worker.translate)
        self.whisper_worker.status_updated.connect(self.update_status)
        self.whisper_worker.start()

        # Поток захвата звука
        self.audio_worker = AudioWorker(device_index, device_sr, self.audio_queue)
        self.audio_worker.start()

    def stop_capture(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.lbl_status.setText("Остановлено.")

        if self.audio_worker:
            self.audio_worker.stop()
            self.audio_worker = None
        if self.whisper_worker:
            self.whisper_worker.stop()
            self.whisper_worker = None
        if self.translation_worker:
            self.translation_worker.stop()
            self.translation_worker = None

    def append_text(self, text):
        self.text_edit.append(text)
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_status(self, text):
        self.lbl_status.setText(text)

    def closeEvent(self, event):
        self.stop_capture()
        self.hebrew_window.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LiveTranslateApp()
    window.show()
    sys.exit(app.exec())
