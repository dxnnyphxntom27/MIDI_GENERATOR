import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget,
    QFileDialog, QLabel, QStackedWidget, QHBoxLayout, QSlider, QSizePolicy,
    QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

import tokenize_user_input
import prompt_generator
import generate_from_input as generatefi
import generate
from datetime import datetime
import clear

SAVE_PATH = Path("app/user_files.json")
TOKENIZED_DIR = Path("app/USER_TOKENIZED")
PROMPT_OUT = Path("app/user_prompt.txt")
GENERATED_MIDI = Path("app/generated.mid")

SLIDER_STYLE = """
    QSlider::groove:horizontal {
        border: none;
        height: 4px;
        background: #394B59;
        margin: 0px 0;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #4361ee;
        border: none;
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }
    QSlider::sub-page:horizontal {
        background: #4869f6;
        border-radius: 2px;
    }
    QSlider::add-page:horizontal {
        background: #222A30;
        border-radius: 2px;
    }
"""

USER_INPUT_SLIDER_STYLE = """
    QSlider::groove:horizontal {
        border: none;
        height: 4px;
        background: #394B59;
        margin: 0px 0;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #7813c6;
        border: none;
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }
    QSlider::sub-page:horizontal {
        background: #7813c6;
        border-radius: 2px;
    }
    QSlider::add-page:horizontal {
        background: #222A30;
        border-radius: 2px;
    }
"""

INSTRUMENTS = [
    (0,  "Acoustic Grand Piano"),
    (1,  "Bright Acoustic Piano"),
    (24, "Acoustic Guitar (steel)"),
    (25, "Acoustic Guitar (nylon)"),
    (27, "Electric Guitar (clean)"),
    (28, "Electric Guitar (muted)"),
    (29, "Overdriven Guitar"),
    (30, "Distortion Guitar"),
]


def generate_filename():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"app/generated_{now}.mid")

class ToggleWidget(QWidget):
    def __init__(self, on_toggle):
        super().__init__()
        layout = QHBoxLayout(self)
        self.toggle_btn = QPushButton("Switch mode")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.clicked.connect(self.on_toggle_clicked)
        self.on_toggle = on_toggle
        layout.addWidget(self.toggle_btn)
        layout.addStretch(1)
        self.setLayout(layout)
        self.update_style()

    def on_toggle_clicked(self):
        idx = 0 if self.toggle_btn.isChecked() else 1
        self.on_toggle(idx)
        self.update_style()

    def set_mode(self, idx):
        self.toggle_btn.setChecked(idx == 0)
        self.update_style()

    def update_style(self):
        if self.toggle_btn.isChecked():
            self.toggle_btn.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #7813c6, stop:1 #6710aa
                    );
                    color: #FFF;
                    border-radius: 14px;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 10px 24px;
                    min-width: 220px;
                }
                QPushButton:hover {
                    background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6710aa, stop:1 #580f91
                );
            }
                """
            )
        else:
            self.toggle_btn.setStyleSheet(
                """
                QPushButton {
                    background: #23292E;
                    color: #EEE;
                    border-radius: 14px;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 10px 24px;
                    min-width: 220px;
                }
                """
            )

class UserInputTab(QWidget):
    def __init__(self, status_label):
        super().__init__()
        self.status_label = status_label

        layout = QVBoxLayout(self)
        layout.setSpacing(18)

        label = QLabel("Select 1-15 MIDI files to inspire the generator with:")
        label.setStyleSheet("font-size: 16px; margin-bottom: 2px; font-weight: 500; color: #9fb8d4;")
        layout.addWidget(label)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background: #1B232B;
                border: none;
                border-radius: 18px;
                padding: 8px 14px;
                min-height: 80px;
                font-size: 15px;
                color: #C2E6F5;
            }
            QListWidget::item:selected {
                background: #4361ee;
                color: #FFF;
                border-radius: 12px;
            }
        """)
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.file_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.file_list)

        # --- Slider długości ---
        len_row = QHBoxLayout()
        len_label = QLabel("Length (tokens):")
        self.len_slider = QSlider(Qt.Orientation.Horizontal)
        self.len_slider.setMinimum(150)
        self.len_slider.setMaximum(500)
        self.len_slider.setSingleStep(10)
        self.len_slider.setTickInterval(10)
        self.len_slider.setValue(200)
        self.len_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.len_slider.setFixedWidth(260)
        self.len_slider.setStyleSheet(USER_INPUT_SLIDER_STYLE)
        self.len_slider.valueChanged.connect(self.update_len_label)
        self.len_display = QLabel(str(self.len_slider.value()))
        self.len_display.setStyleSheet("color: #A0AAB8; font-weight: bold; margin-left: 10px;")
        len_row.addWidget(len_label)
        len_row.addWidget(self.len_slider)
        len_row.addWidget(self.len_display)
        len_row.addStretch(1)
        layout.addLayout(len_row)

        btn_row = QHBoxLayout()
        btn_select = QPushButton("Select MIDI files")
        btn_select.clicked.connect(self.choose_files)
        btn_select.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d6d6d, stop:1 #4a4a4a
                );
                color: #FFF;
                border-radius: 14px;
                font-weight: 500;
                font-size: 15px;
                padding: 10px 30px;
                margin-right: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5d5d5d, stop:1 #3c3c3c
                );
            }
        """)
        btn_row.addWidget(btn_select)

        btn_generate = QPushButton("GENERATE MUSIC")
        btn_generate.clicked.connect(self.full_pipeline)
        btn_generate.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7813c6, stop:1 #6710aa
                );
                color: #FFF;
                border-radius: 14px;
                font-weight: bold;
                font-size: 15px;
                padding: 10px 30px;                 
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6710aa, stop:1 #580f91
                );
            }
        """)
        btn_row.addWidget(btn_generate)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def update_len_label(self, value):
        snapped = int(round(value / 10) * 10)
        self.len_slider.blockSignals(True)
        self.len_slider.setValue(snapped)
        self.len_slider.blockSignals(False)
        self.len_display.setText(str(snapped))

    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select MIDI files",
            "",
            "MIDI Files (*.mid *.midi)"
        )
        if not files:
            return
        if len(files) > 15:
            self.status_label.setText("⚠️ Select up to 15 files only!")
            return
        self.file_list.clear()
        self.file_list.addItems(files)
        self.status_label.setText(f"Selected {len(files)} file(s).")

    def full_pipeline(self):
        paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not (1 <= len(paths) <= 15):
            self.status_label.setText("⚠️ Select 1-15 files only!")
            return

        max_tokens = self.len_slider.value()
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=2, ensure_ascii=False)
        self.status_label.setText(f"✅ Saved {len(paths)} path(s). Tokenizing...")
        QApplication.processEvents()

        try:
            tokenize_user_input.tokenize_files(
                midi_files=paths,
                output_dir=TOKENIZED_DIR
            )
            self.status_label.setText("✅ Tokenization complete. Generating prompt...")
            QApplication.processEvents()
        except Exception as e:
            self.status_label.setText(f"❌ Tokenization failed: {e}")
            return

        mode = "single" if len(paths) == 1 else "multi"
        try:
            prompt_generator.generate_prompt(
                token_dir=TOKENIZED_DIR,
                prompt_out=PROMPT_OUT,
                tokens_after_meta=16,
                max_tokens=64,
                random_instrument=False,
                mode=mode
            )
            self.status_label.setText(f"✅ Prompt ready. Generating music...")
            QApplication.processEvents()
        except Exception as e:
            self.status_label.setText(f"❌ Prompt generation failed: {e}")
            return

        def read_prompt_tokens(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        try:
            start_tokens = read_prompt_tokens(PROMPT_OUT)
            midi_path = generatefi.generate_midi_from_prompt(
                start_tokens=start_tokens,
                max_tokens=max_tokens,
                output_midi=generate_filename(),
                temperature=1.0,
                top_k=50,
                stop_token="<SONG_END>",
                max_tries=10,
                verbose=True
            )
            if midi_path:
                self.status_label.setText(f"🎵 DONE! MIDI file saved as: {midi_path}")
                clear.clear_user_artifacts(tokenized_dir=TOKENIZED_DIR, prompt_file=PROMPT_OUT)
            else:
                self.status_label.setText("❌ Generation failed (model didn't meet musical requirements).")
        except Exception as e:
            self.status_label.setText(f"❌ MIDI generation failed: {e}")

class StandardGenTab(QWidget):
    def __init__(self, status_label):
        super().__init__()
        self.status_label = status_label

        layout = QVBoxLayout(self)
        label = QLabel("Standard generation (no user MIDI input).")
        label.setStyleSheet("font-size: 16px; margin-bottom: 16px; color: #9fb8d4; font-weight: 500;")
        layout.addWidget(label)

        # TEMPO + INSTRUMENT w jednym rzędzie
        param_row = QHBoxLayout()
        
        # TEMPO SLIDER
        tempo_label = QLabel("Tempo (BPM):")
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setMinimum(70)
        self.tempo_slider.setMaximum(185)
        self.tempo_slider.setSingleStep(5)
        self.tempo_slider.setTickInterval(5)
        self.tempo_slider.setValue(120)
        self.tempo_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tempo_slider.setFixedWidth(170)
        self.tempo_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: none; height: 4px; background: #394B59; margin: 0px 0; border-radius: 2px; }
            QSlider::handle:horizontal { background: #4361ee; border: none; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
            QSlider::sub-page:horizontal { background: #4869f6; border-radius: 2px; }
            QSlider::add-page:horizontal { background: #222A30; border-radius: 2px; }
        """)
        self.tempo_slider.valueChanged.connect(self.update_tempo_label)
        self.tempo_display = QLabel(str(self.tempo_slider.value()))
        self.tempo_display.setStyleSheet("color: #A0AAB8; font-weight: bold; margin-left: 10px; margin-right: 20px;")
        param_row.addWidget(tempo_label)
        param_row.addWidget(self.tempo_slider)
        param_row.addWidget(self.tempo_display)

        # INSTRUMENT DROPDOWN
        instrument_label = QLabel("Instrument:")
        self.instrument_dropdown = QComboBox()
        for num, name in INSTRUMENTS:
            self.instrument_dropdown.addItem(name, num)
        self.instrument_dropdown.setCurrentIndex(0)
        self.instrument_dropdown.setStyleSheet(
            "background: #222A30; color: #EEE; border-radius: 8px; padding: 2px 8px; font-size: 15px;"
        )
        param_row.addWidget(instrument_label)
        param_row.addWidget(self.instrument_dropdown)
        param_row.addStretch(1)
        layout.addLayout(param_row)

        # --- DŁUGOŚĆ ---
        len_row = QHBoxLayout()
        len_label = QLabel("Song length (tokens):")
        self.len_slider = QSlider(Qt.Orientation.Horizontal)
        self.len_slider.setMinimum(150)
        self.len_slider.setMaximum(500)
        self.len_slider.setSingleStep(10)
        self.len_slider.setTickInterval(10)
        self.len_slider.setValue(200)
        self.len_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.len_slider.setFixedWidth(260)
        self.len_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: none; height: 4px; background: #394B59; margin: 0px 0; border-radius: 2px; }
            QSlider::handle:horizontal { background: #4361ee; border: none; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
            QSlider::sub-page:horizontal { background: #4869f6; border-radius: 2px; }
            QSlider::add-page:horizontal { background: #222A30; border-radius: 2px; }
        """)
        self.len_slider.valueChanged.connect(self.update_len_label)
        self.len_display = QLabel(str(self.len_slider.value()))
        self.len_display.setStyleSheet("color: #A0AAB8; font-weight: bold; margin-left: 10px;")
        len_row.addWidget(len_label)
        len_row.addWidget(self.len_slider)
        len_row.addWidget(self.len_display)
        len_row.addStretch(1)
        layout.addLayout(len_row)

        btn_generate = QPushButton("GENERATE STANDARD MIDI")
        btn_generate.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4361ee, stop:1 #4f8aff
                );
                color: #FFF;
                border-radius: 14px;
                font-weight: bold;
                font-size: 15px;
                padding: 10px 30px;
                margin-top: 16px;
            }
            QPushButton:hover {
                background: #2950d9;
            }
            """
        )
        btn_generate.clicked.connect(self.generate_standard)
        layout.addWidget(btn_generate)
        self.setLayout(layout)

    def update_tempo_label(self, value):
        snapped = int(round(value / 5) * 5)
        self.tempo_slider.blockSignals(True)
        self.tempo_slider.setValue(snapped)
        self.tempo_slider.blockSignals(False)
        self.tempo_display.setText(str(snapped))

    def update_len_label(self, value):
        snapped = int(round(value / 10) * 10)
        self.len_slider.blockSignals(True)
        self.len_slider.setValue(snapped)
        self.len_slider.blockSignals(False)
        self.len_display.setText(str(snapped))

    def generate_standard(self):
        tempo = self.tempo_slider.value()
        tempo = int(round(tempo / 5) * 5)
        max_tokens = self.len_slider.value()
        max_tokens = int(round(max_tokens / 10) * 10)
        instr_num = int(self.instrument_dropdown.currentData())
        start_tokens = [
            "key_unknown",
            f"tempo_{tempo}",
            "track_0",
            f"instrument_{instr_num}"
        ]
        self.status_label.setText(f"⏳ Generating with tempo {tempo} BPM, instrument {instr_num}...")
        QApplication.processEvents()
        try:
            import generate
            midi_path = generate.generate_midi_from_prompt(
                start_tokens=start_tokens,
                max_tokens=max_tokens,
                output_midi=generate_filename(),
                temperature=1.0,
                top_k=50,
                stop_token="<SONG_END>",
                max_tries=10,
                verbose=True
            )
            if midi_path:
                self.status_label.setText(f"🎵 DONE! MIDI file saved as: {midi_path}")
            else:
                self.status_label.setText("❌ Generation failed (model didn't meet musical requirements).")
        except Exception as e:
            self.status_label.setText(f"❌ MIDI generation failed: {e}")


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Generator")
        self.setFixedWidth(700)
        self.setStyleSheet("""
            QWidget { background: #181C1F; color: #ECECEC; font-size: 16px; }
            QPushButton { background: #222A30; color: #EEE; border-radius: 14px; padding: 10px; }
            QPushButton:hover { background: #3D4852; }
            QListWidget { background: #1B232B; border-radius: 18px; padding: 8px; }
            QLabel { color: #B0BEC5; }
            QSlider { background: transparent; }
        """)

        main_layout = QVBoxLayout(self)
        self.toggle = ToggleWidget(self.switch_mode)
        main_layout.addWidget(self.toggle)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("margin: 14px 0 0 0; font-size: 15px; color: #A0AAB8;")

        self.stacked = QStackedWidget()
        self.user_tab = UserInputTab(self.status_label)
        self.std_tab = StandardGenTab(self.status_label)
        self.stacked.addWidget(self.user_tab)
        self.stacked.addWidget(self.std_tab)
        main_layout.addWidget(self.stacked)
        main_layout.addWidget(self.status_label)
        self.setLayout(main_layout)

    def switch_mode(self, idx):
        self.stacked.setCurrentIndex(idx)
        self.toggle.set_mode(idx)
        self.status_label.setText("")

    def closeEvent(self, event):
        clear.clear_user_artifacts(tokenized_dir=TOKENIZED_DIR, prompt_file=PROMPT_OUT)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.setWindowIcon(QIcon("app/icon.png"))
    win.show()
    sys.exit(app.exec())
