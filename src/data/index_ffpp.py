from pathlib import Path
import csv
import random

# Raíz REAL donde están original, Deepfakes, Face2Face, etc.
FFPP_ROOT = Path("data/raw/ffpp/FaceForensics++_C23")

OUTPUT_CSV = Path("data/ffpp_videos.csv")

FAKE_FOLDERS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
]

IGNORE_FOLDERS = [
    "csv",
    "DeepFakeDetection",
]

def main():
    video_rows = []

    for video_path in FFPP_ROOT.rglob("*.mp4"):
        rel = video_path.relative_to(FFPP_ROOT)
        parts = rel.parts

        # Esperamos algo tipo:
        # original/000.mp4
        # NeuralTextures/989_993.mp4
        if len(parts) < 2:
            print(f"[WARN] Ruta inesperada: {rel}")
            continue

        top_folder = parts[0]  # original / Deepfakes / Face2Face / ...

        if top_folder in IGNORE_FOLDERS:
            continue

        if top_folder == "original":
            label = 0
            class_name = "real"
            manipulation_type = "none"
        elif top_folder in FAKE_FOLDERS:
            label = 1
            class_name = "fake"
            manipulation_type = top_folder
        else:
            print(f"[WARN] Carpeta desconocida (top_folder={top_folder}) para {video_path}")
            continue

        video_rows.append({
            "video_path": video_path.as_posix(),
            "label": label,
            "class_name": class_name,
            "manipulation_type": manipulation_type,
        })

    if not video_rows:
        print("⚠️ No se encontraron videos válidos. Revisa FFPP_ROOT.")
        return

    # Mezclar y hacer splits 70/15/15
    random.seed(42)
    random.shuffle(video_rows)
    n = len(video_rows)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    for i, row in enumerate(video_rows):
        if i < n_train:
            row["split"] = "train"
        elif i < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "label", "class_name", "manipulation_type", "split"],
        )
        writer.writeheader()
        writer.writerows(video_rows)

    print(f"✅ Videos indexados: {len(video_rows)}")
    print(f"✅ CSV guardado en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
