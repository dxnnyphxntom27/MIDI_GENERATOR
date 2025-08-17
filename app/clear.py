import shutil
from pathlib import Path

def clear_user_artifacts(
    tokenized_dir="app/USER_TOKENIZED",
    prompt_file="app/user_prompt.txt"
):
    tokenized_path = Path(tokenized_dir)
    prompt_path = Path(prompt_file)
    # Usuń folder USER_TOKENIZED (rekursywnie)
    if tokenized_path.exists() and tokenized_path.is_dir():
        try:
            shutil.rmtree(tokenized_path)
            print(f"🗑️ Usunięto folder: {tokenized_path}")
        except Exception as e:
            print(f"⚠️ Błąd podczas usuwania {tokenized_path}: {e}")
    # Usuń plik promptu
    if prompt_path.exists() and prompt_path.is_file():
        try:
            prompt_path.unlink()
            print(f"🗑️ Usunięto plik: {prompt_path}")
        except Exception as e:
            print(f"⚠️ Błąd podczas usuwania {prompt_path}: {e}")
