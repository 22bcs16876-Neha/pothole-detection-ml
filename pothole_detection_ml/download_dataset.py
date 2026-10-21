"""
Download public pothole images and add them to dataset/pothole/.

Run from the project folder:
    python download_dataset.py

Requires an internet connection. Images are taken from the public release
linked in the dataset publisher's GitHub repository.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile
from io import BytesIO
import shutil
import sys

URL = (
    "https://github.com/jaygala24/pothole-detection/releases/download/"
    "v1.0.0/Pothole.Dataset.IVCNZ.zip"
)
OUT = Path("dataset") / "pothole"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Downloading public pothole image dataset...")
    print("This needs an internet connection and may take a few minutes.")
    request = Request(URL, headers={"User-Agent": "PotholeDetectionStudentProject/1.0"})
    try:
        with urlopen(request, timeout=90) as response:
            archive_bytes = response.read()
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("Check your internet connection, then run this command again.")
        sys.exit(1)

    try:
        archive = ZipFile(BytesIO(archive_bytes))
    except BadZipFile:
        print("Downloaded file was not a valid ZIP archive.")
        sys.exit(1)

    copied = 0
    skipped = 0
    for member in archive.infolist():
        if member.is_dir():
            continue
        name = Path(member.filename)
        if name.suffix.lower() not in IMAGE_EXTS:
            continue
        # Avoid collisions and path traversal; use the basename only.
        safe_name = f"public_pothole_{copied + 1:04d}{name.suffix.lower()}"
        destination = OUT / safe_name
        try:
            with archive.open(member) as file_in, destination.open("wb") as file_out:
                shutil.copyfileobj(file_in, file_out)
            # Verify image bytes with Pillow if available.
            try:
                from PIL import Image
                with Image.open(destination) as image:
                    image.verify()
            except Exception:
                destination.unlink(missing_ok=True)
                skipped += 1
                continue
            copied += 1
        except Exception:
            destination.unlink(missing_ok=True)
            skipped += 1

    print(f"\nAdded {copied} pothole images to: {OUT}")
    if skipped:
        print(f"Skipped {skipped} unreadable image files.")
    if copied == 0:
        print("No images were extracted. Check the source archive and internet access.")
        sys.exit(1)
    print("\nNext step: python train.py")
    print("Important: visually review labels and dataset quality before trusting metrics.")

if __name__ == "__main__":
    main()
