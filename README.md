# LSTM MIDI Generator

An AI-driven music generation tool that uses Long Short-Term Memory (LSTM) neural networks to synthesize MIDI files.

## Features

The application operates in two distinct generation modes:
* **Mode 1 (Seed-Based):** Generates new musical sequences based on user-provided `.mid` files. The model analyzes the input pattern and extends it while maintaining the original stylistic context.
* <img width="697" height="473" alt="image" src="https://github.com/user-attachments/assets/5c109856-a5e1-44ac-b635-2aa30c16fd1e" />
* **Mode 2 (Instrument-Based):** Generates original MIDI sequences tailored to a specific, user-selected instrument profile from scratch.
* <img width="694" height="479" alt="image" src="https://github.com/user-attachments/assets/82f767bf-cd75-4668-88d0-50137776ff80" />


## Tech Stack
* **Language:** Python
* **Machine Learning:** PyTorch
* **Data Processing:** `mido` / `pretty_midi`
* **Visu:**: PyQt6
