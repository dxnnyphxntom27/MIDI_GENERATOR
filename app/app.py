import sys
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QFileDialog, QLabel
)
from PyQt6.QtCore import Qt

import tokenize_user_input
import prompt_generator
import generate
import clear  # <-- import czyszczenia!

SAVE_PATH = Path("app/user_files.json")
TOKENIZED_DIR = Path("app/USER_TOKENIZED")
PROMPT_OUT = Path("app/user_prompt.txt")
GENERATED_MIDI = Path("app/generated.mid")

class MidiFileSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Generator")
        self.setFixedWidth(620)
        self.setStyleSheet("""
            QWidget { background: #181C1F; color: #ECECEC; font-size: 16px; }
            QPushButton { background: #222A30; color: #EEE; border-radius: 14px; padding: 10px; }
            QPushButton:hover { background: #3D4852; }
            QListWidget { background: #222A30; border-radius: 14px; padding: 8px; }
            QLabel { color: #B0BEC5; }
        """)

        layout = QVBoxLayout(self)
        label = QLabel("Select 1-15 MIDI files to inspire the generator with:")
        layout.addWidget(label)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.file_list)

        btn_select = QPushButton("Select MIDI files")
        btn_select.clicked.connect(self.choose_files)
        layout.addWidget(btn_select)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btn_generate = QPushButton("GENERATE MUSIC")
        btn_generate.clicked.connect(self.full_pipeline)
        layout.addWidget(btn_generate)

        self.setLayout(layout)

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

        # 1. Save file list to user_files.json
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=2, ensure_ascii=False)
        self.status_label.setText(f"✅ Saved {len(paths)} path(s). Tokenizing...")
        QApplication.processEvents()

        # 2. Tokenize
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

        # 3. Prompt
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

        # 4. Read prompt tokens
        def read_prompt_tokens(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        try:
            start_tokens = read_prompt_tokens(PROMPT_OUT)
            midi_path = generate.generate_midi_from_prompt(
                start_tokens=start_tokens,
                max_tokens=100,
                output_midi=GENERATED_MIDI,
                temperature=1.0,
                top_k=50,
                stop_token="<SONG_END>",
                max_tries=10,
                verbose=True
            )
            if midi_path:
                self.status_label.setText(f"🎵 DONE! MIDI file saved as: {midi_path}")
                # Po generacji – czyść katalog/plik
                clear.clear_user_artifacts(tokenized_dir=TOKENIZED_DIR, prompt_file=PROMPT_OUT)
            else:
                self.status_label.setText("❌ Generation failed (model didn't meet musical requirements).")
        except Exception as e:
            self.status_label.setText(f"❌ MIDI generation failed: {e}")

    def closeEvent(self, event):
        # Usuwaj artefakty także przy zamykaniu aplikacji
        clear.clear_user_artifacts(tokenized_dir=TOKENIZED_DIR, prompt_file=PROMPT_OUT)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MidiFileSelector()
    win.show()
    sys.exit(app.exec())
