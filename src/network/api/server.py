import os
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile

from auth.face import FaceVerification
from auth.voice import VoiceAuth


class FileUploadAPI():
    def __init__(self, face_verification_config, voice_auth_config):
        self.app = FastAPI()
        self.setup_routes()
        self.face_verification = FaceVerification(face_verification_config)
        self.voice_auth = VoiceAuth(voice_auth_config)

    def setup_routes(self):
        @self.app.post("/upload/")
        async def upload_files(
                username: str,
                wav_files: List[UploadFile] = File(...),
                jpg_file: UploadFile = File(...)
        ):
            return await self.handle_upload(username, wav_files, jpg_file)

    async def handle_upload(self, username: str, wav_files: List[UploadFile], jpg_file: UploadFile):
        save_path = "uploads"
        os.makedirs(save_path, exist_ok=True)
        jpg_path = os.path.join(save_path, jpg_file.filename)
        wav_files_path = []
        for wav_file in wav_files:
            wav_files_path.append(os.path.join(save_path, wav_file.filename))
            with open(os.path.join(save_path, wav_file.filename), "wb") as buffer:
                buffer.write(wav_file.file.read())
        with open(jpg_path, "wb") as buffer:
            buffer.write(jpg_file.file.read())

        if not self.face_verification.build_embedding_from_image(jpg_path, username):
            return {"message": "Failed to build embedding from image"}
        if not self.voice_auth.enroll_user(username, wav_files_path):
            return {"message": "Failed to build embedding from voice"}

        return {"message": "Upload successful!"}


if __name__ == "__main__":
    face_verification_config = {
        "model_path": "mobilefacenet.tflite",
        "database_path": "face_embeddings.pkl",
        # Face detection & preprocessing settings:
        "model_selection": 0,
        "min_detection_confidence": 0.7,
        "padding": 0.2,
        "face_required_size": (512, 512),
        "target_size": (112, 112),
        # Verification settings:
        "verification_threshold": 0.7,
        "verification_timeout": 30,
        "verification_max_attempts": 3,
        # Camera settings:
        "camera_id": 0,
    }

    voice_auth_config = {
        "voiceprints_file": "voiceprints.pkl",
        "sr_rate": 44100,
        "num_mfcc": 20,
        "linear_threshold": 100,
        "cos_threshold": 0.95,
    }
    file_upload_api = FileUploadAPI(face_verification_config, voice_auth_config)
    uvicorn.run(file_upload_api.app, host="0.0.0.0", port=8000)
