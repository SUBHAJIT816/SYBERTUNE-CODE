import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import bessel, sosfilt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSlider, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QPushButton, 
                             QFileDialog, QFrame, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QFont

# --- Audio Constants ---
BLOCK_SIZE = 2048
BANDS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

# --- Presets Data ---
PRESETS = {
    "Flat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Rock": [5, 4, 3, 0, -1, -1, 0, 3, 4, 5],
    "Pop": [-2, -1, 0, 2, 4, 4, 2, 0, -1, -2],
    "Jazz": [4, 3, 1, 2, -2, -2, 0, 1, 3, 4],
    "Classical": [5, 4, 3, 2, -1, -1, 0, 2, 4, 5],
    "Electronic": [6, 5, 0, -2, 2, 0, 4, 5, 6, 6],
    "Vocal": [-3, -2, -1, 1, 3, 4, 4, 3, 1, -1]
}

class AudioWorker(QObject):
    spectrum_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.fs = 44100
        self.current_frame = 0
        self.is_playing = False
        self.gains = np.zeros(len(BANDS))
        self.preamp = 1.0
        self.master_vol = 0.8
        self.bass_boost = 0.0  
        self.treble_boost = 0.0 
        self.is_bass_booster_active = False
        self.sos_filters = []

    def load_file(self, path):
        self.data, self.fs = sf.read(path, always_2d=True)
        self.current_frame = 0
        self._update_filters()

    def _update_filters(self):
        self.sos_filters = []
        for freq in BANDS:
            low = freq * 0.707
            high = min(freq * 1.414, self.fs / 2 - 1)
            sos = bessel(2, [low, high], btype='bandpass', fs=self.fs, output='sos')
            self.sos_filters.append(sos)

    def callback(self, outdata, frames, time, status):
        if not self.is_playing or self.data is None:
            outdata.fill(0)
            return

        chunk_end = self.current_frame + frames
        if chunk_end > len(self.data):
            self.is_playing = False
            return

        chunk = self.data[self.current_frame:chunk_end].copy()
        chunk *= self.preamp

        processed_chunk = chunk.copy()
        
        for i, sos in enumerate(self.sos_filters):
            total_gain = self.gains[i]
            if i <= 2:
                total_gain += self.bass_boost
                if self.is_bass_booster_active:
                    total_gain += 8.0
            if i >= 7:
                total_gain += self.treble_boost

            if total_gain != 0:
                filtered = sosfilt(sos, chunk, axis=0)
                linear_gain = 10**(total_gain/20) - 1
                processed_chunk += filtered * linear_gain

        processed_chunk *= self.master_vol
        np.clip(processed_chunk, -1, 1, outdata)

        avg_signal = np.mean(processed_chunk, axis=1)
        fft_data = np.abs(np.fft.rfft(avg_signal * np.hanning(len(avg_signal))))
        
        # Safe check before emitting to avoid RuntimeError on shutdown
        try:
            self.spectrum_ready.emit(fft_data)
        except RuntimeError:
            pass
            
        self.current_frame += frames

class Visualizer(QFrame):
    def __init__(self):
        super().__init__()
        self.spectrum = np.zeros(60)
        self.setMinimumHeight(180)
        self.setStyleSheet("background-color: #1a1e2e; border-radius: 15px; border: 1px solid #2a314d;")

    def update_spectrum(self, data):
        samples = 60
        if len(data) > samples:
            indices = np.linspace(0, len(data)-1, samples).astype(int)
            self.spectrum = data[indices]
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bar_w = w / len(self.spectrum)

        for i, val in enumerate(self.spectrum):
            bar_h = min(h - 20, (np.log10(val + 1) * h * 0.6))
            grad = QLinearGradient(0, h, 0, h - bar_h)
            grad.setColorAt(0, QColor(112, 0, 255, 180)) # Purple
            grad.setColorAt(1, QColor(0, 212, 255))      # Cyan
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(int(i * bar_w + 2), int(h - bar_h - 10), int(bar_w - 4), int(bar_h), 3, 3)

class EqualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = AudioWorker(self)
        self.band_sliders = []
        self.init_ui()
        
        self.stream = sd.OutputStream(
            samplerate=44100, channels=2, blocksize=BLOCK_SIZE, 
            callback=self.worker.callback
        )
        self.stream.start()

    def init_ui(self):
        self.setWindowTitle("CyberTune | Pro Audio EQ")
        self.setMinimumSize(1100, 800)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #a1a1a1; font-family: 'Segoe UI'; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; }
            
            QComboBox {
                background-color: #161b22; color: #58a6ff; border: 1px solid #30363d;
                padding: 5px 10px; border-radius: 6px; min-width: 150px; font-weight: bold;
            }
            
            QCheckBox { color: #58a6ff; font-weight: bold; spacing: 10px; }
            QCheckBox::indicator { width: 20px; height: 20px; border-radius: 6px; border: 2px solid #30363d; background: #161b22; }
            QCheckBox::indicator:checked { background: #58a6ff; border-color: #58a6ff; }

            QPushButton { 
                background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; 
                border-radius: 6px; padding: 10px 25px; font-weight: bold;
            }
            QPushButton:hover { background-color: #30363d; color: #ffffff; border-color: #8b949e; }
            QPushButton#play { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1f6feb, stop:1 #388bfd); 
                color: white; border: none; 
            }
            QPushButton#play:hover { background: #388bfd; }

            QSlider::groove:vertical { background: #21262d; width: 6px; border-radius: 3px; }
            QSlider::handle:vertical { 
                background: #58a6ff; border: 2px solid #ffffff; height: 18px; width: 18px; 
                margin: 0 -6px; border-radius: 9px; 
            }
            QSlider::groove:horizontal { background: #21262d; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { 
                background: #58a6ff; border: 2px solid #ffffff; height: 18px; width: 18px; 
                margin: -6px 0; border-radius: 9px; 
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        # 1. Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        main_title = QLabel("CYBERTUNE")
        main_title.setStyleSheet("font-size: 28px; font-weight: 800; color: #ffffff; letter-spacing: 6px;")
        self.file_label = QLabel("SYSTEM ONLINE • NO SOURCE")
        self.file_label.setStyleSheet("color: #58a6ff; font-weight: bold;")
        title_box.addWidget(main_title)
        title_box.addWidget(self.file_label)
        header.addLayout(title_box)
        header.addStretch()
        
        self.style_combo = QComboBox()
        self.style_combo.addItems(PRESETS.keys())
        self.style_combo.currentTextChanged.connect(self.apply_preset)
        header.addWidget(QLabel("Profile"))
        header.addWidget(self.style_combo)
        
        btn_load = QPushButton("IMPORT")
        btn_load.clicked.connect(self.load_audio)
        self.btn_play = QPushButton("START ENGINE")
        self.btn_play.setObjectName("play")
        self.btn_play.clicked.connect(self.toggle_play)
        header.addWidget(btn_load)
        header.addWidget(self.btn_play)
        layout.addLayout(header)

        # 2. Visualizer
        self.viz = Visualizer()
        self.worker.spectrum_ready.connect(self.viz.update_spectrum)
        layout.addWidget(self.viz)

        # 3. Controls Area
        main_controls = QHBoxLayout()
        
        # Side Panel (Bass/Treble)
        side_panel = QFrame()
        side_panel.setStyleSheet("background-color: #161b22; border-radius: 15px; border: 1px solid #30363d;")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(15, 20, 15, 20)

        self.bass_slider = QSlider(Qt.Orientation.Vertical)
        self.bass_slider.setRange(0, 15)
        self.bass_slider.valueChanged.connect(lambda v: setattr(self.worker, 'bass_boost', v))
        
        self.treble_slider = QSlider(Qt.Orientation.Vertical)
        self.treble_slider.setRange(0, 15)
        self.treble_slider.valueChanged.connect(lambda v: setattr(self.worker, 'treble_boost', v))
        
        self.boost_chk = QCheckBox("BOOST")
        self.boost_chk.stateChanged.connect(self.toggle_bass_booster)

        side_layout.addWidget(QLabel("Bass"), alignment=Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(self.bass_slider, alignment=Qt.AlignmentFlag.AlignCenter)
        side_layout.addSpacing(15)
        side_layout.addWidget(QLabel("Treble"), alignment=Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(self.treble_slider, alignment=Qt.AlignmentFlag.AlignCenter)
        side_layout.addSpacing(15)
        side_layout.addWidget(self.boost_chk, alignment=Qt.AlignmentFlag.AlignCenter)
        
        main_controls.addWidget(side_panel)

        # EQ Main Card
        eq_card = QFrame()
        eq_card.setStyleSheet("background-color: #161b22; border-radius: 20px; border: 1px solid #30363d;")
        eq_layout = QHBoxLayout(eq_card)
        eq_layout.setContentsMargins(25, 35, 25, 35)
        
        preamp_box = self.create_slider_unit("PRE", 0, 100, 50, self.update_preamp)
        eq_layout.addLayout(preamp_box)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("background-color: #30363d; max-width: 1px;")
        eq_layout.addWidget(line)

        for i, freq in enumerate(BANDS):
            label = f"{freq if freq < 1000 else f'{freq//1000}K'}"
            unit = self.create_slider_unit(label, -15, 15, 0, 
                                          lambda v, idx=i: self.update_band(idx, v))
            eq_layout.addLayout(unit)
            self.band_sliders.append(unit.itemAt(0).widget())
        
        main_controls.addWidget(eq_card)
        layout.addLayout(main_controls)

        # 4. Footer
        footer = QHBoxLayout()
        master_vol_box = self.create_slider_unit("OUTPUT", 0, 100, 80, self.update_master, horizontal=True)
        footer.addLayout(master_vol_box)
        footer.addStretch()
        footer.addWidget(QLabel("STUDIO QUALITY DSP • LOW LATENCY ENGINE"))
        layout.addLayout(footer)

    def create_slider_unit(self, label_text, min_v, max_v, def_v, callback, horizontal=False):
        layout = QVBoxLayout() if not horizontal else QHBoxLayout()
        slider = QSlider(Qt.Orientation.Vertical if not horizontal else Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(def_v)
        slider.setMinimumHeight(180) if not horizontal else slider.setFixedWidth(250)
        slider.valueChanged.connect(callback)
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(slider, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        return layout

    def apply_preset(self, name):
        if name in PRESETS:
            preset_gains = PRESETS[name]
            for i, gain in enumerate(preset_gains):
                self.band_sliders[i].setValue(gain)
                self.worker.gains[i] = gain

    def toggle_bass_booster(self, state):
        self.worker.is_bass_booster_active = (state == 2)

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio (*.wav *.flac *.mp3)")
        if path:
            self.worker.load_file(path)
            self.file_label.setText(f"SOURCE: {path.split('/')[-1].upper()}")

    def toggle_play(self):
        if self.worker.data is None: return
        self.worker.is_playing = not self.worker.is_playing
        self.btn_play.setText("PAUSE ENGINE" if self.worker.is_playing else "START ENGINE")

    def update_preamp(self, val): self.worker.preamp = val / 50.0
    def update_master(self, val): self.worker.master_vol = val / 100.0
    def update_band(self, index, val): self.worker.gains[index] = val

    def closeEvent(self, event):
        """Cleanup audio stream before closing the window."""
        self.worker.is_playing = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EqualizerApp()
    window.show()
    sys.exit(app.exec())