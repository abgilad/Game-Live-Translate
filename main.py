import sys
import queue
import numpy as np
import pyaudiowpatch as pyaudio
from scipy import signal as scipy_signal
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QComboBox, QTextEdit, QLabel, QPushButton,
                             QHBoxLayout, QSlider)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QTextOption

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
# Hebrew Translation Window (subtitle-style, transparent)
# ---------------------------------------------------------
class HebrewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("תרגום לעברית")
        self.resize(900, 220)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_pos = None
        self._bg_alpha = 160  # default background opacity

        central_widget = QWidget()
        central_widget.setObjectName("subtitle_bg")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # ---- Control bar (drag + opacity slider + close) ----
        ctrl = QHBoxLayout()

        drag_lbl = QLabel("⠿ תרגום לעברית")
        drag_lbl.setStyleSheet("color: rgba(255,255,255,160); font-size: 11px;")
        ctrl.addWidget(drag_lbl)
        ctrl.addStretch()

        opacity_lbl = QLabel("רקע:")
        opacity_lbl.setStyleSheet("color: rgba(255,255,255,160); font-size: 11px;")
        ctrl.addWidget(opacity_lbl)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(self._bg_alpha)
        self.opacity_slider.setFixedWidth(100)
        self.opacity_slider.valueChanged.connect(self._update_bg)
        ctrl.addWidget(self.opacity_slider)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,40); color: white; "
            "border: none; border-radius: 11px; font-size: 12px; }"
            "QPushButton:hover { background: rgba(200,50,50,200); }"
        )
        close_btn.clicked.connect(self.close)
        ctrl.addWidget(close_btn)

        layout.addLayout(ctrl)

        # ---- Hebrew text area ----
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.text_edit.setStyleSheet(
            "background: transparent; border: none; color: white;"
        )
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        font = self.text_edit.font()
        font.setPointSize(32)
        self.text_edit.setFont(font)
        layout.addWidget(self.text_edit)

        self._update_bg(self._bg_alpha)

    def _update_bg(self, value):
        self._bg_alpha = value
        self.centralWidget().setStyleSheet(
            f"QWidget#subtitle_bg {{"
            f"  background-color: rgba(0, 0, 0, {value});"
            f"  border-radius: 14px;"
            f"}}"
        )

    # ---- Drag support ----
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def append_text(self, text):
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.text_edit.append(
            f'<p dir="rtl" align="center" style="font-size:32pt; color:white;">{safe}</p>'
        )
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


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
        self.hebrew_window.text_edit.clear()

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
