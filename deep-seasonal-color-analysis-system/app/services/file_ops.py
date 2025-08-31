import tempfile
import shutil
from typing import List
from fastapi import UploadFile

def save_temp_image(file: UploadFile) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    shutil.copyfileobj(file.file, tmp)
    return tmp.name

def save_multiple_images(files: List[UploadFile]) -> List[str]:
    return [save_temp_image(f) for f in files]
