from pathlib import Path
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import csv
import pandas as pd
from tqdm import tqdm

# CSV que acabamos de generar
FFPP_CSV = Path("data/ffpp_videos.csv")

# Carpeta de salida donde dejaremos las caras recortadas
OUTPUT_ROOT = Path("data/processed/ffpp")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Cuántos frames tomar por video (ajusta según espacio en disco)
FRAMES_PER_VIDEO = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando device:", device)
mtcnn = MTCNN(keep_all=False, device=device)


def extract_frames_from_video(video_path: Path, num_frames: int = FRAMES_PER_VIDEO):
    """Devuelve una lista de (frame_idx, PIL.Image) tomados a intervalos regulares."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir el video: {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        print(f"[WARN] Frame count inválido en: {video_path}")
        return []

    step = max(1, frame_count // num_frames)
    frames = []
    idx = 0

    while len(frames) < num_frames and idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frames.append((idx, img))

        idx += step

    cap.release()
    return frames


def main():
    df = pd.read_csv(FFPP_CSV)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando videos FF++"):
        video_path = Path(row["video_path"])
        label = int(row["label"])
        class_name = row["class_name"]   # "real" / "fake"
        split = row["split"]             # "train" / "val" / "test"
        manipulation_type = row["manipulation_type"]

        out_dir = OUTPUT_ROOT / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = extract_frames_from_video(video_path)

        for frame_idx, img in frames:
            # detectar cara principal
            boxes, probs = mtcnn.detect(img)

            if boxes is None or len(boxes) == 0:
                # No se encontró cara en este frame
                continue

            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face = img.crop((x1, y1, x2, y2))
            face = face.resize((224, 224))

            video_stem = video_path.stem
            out_name = f"{video_stem}_f{frame_idx}_face.jpg"
            out_path = out_dir / out_name

            face.save(out_path, format="JPEG", quality=95)

            rows.append({
                "file_path": out_path.as_posix(),
                "label": label,
                "class_name": class_name,
                "source_dataset": "FFPP",
                "split": split,
                "manipulation_type": manipulation_type,
                "video_path": video_path.as_posix(),
                "frame_idx": frame_idx,
            })

    # Guardar metadata de imágenes
    meta_path = OUTPUT_ROOT / "ffpp_images_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "label",
                "class_name",
                "source_dataset",
                "split",
                "manipulation_type",
                "video_path",
                "frame_idx",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Caras procesadas: {len(rows)}")
    print(f"✅ Metadata de imágenes guardada en: {meta_path}")


if __name__ == "__main__":
    main()
