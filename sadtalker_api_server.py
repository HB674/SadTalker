# sadtalker_api_server.py
import os
import time
from pathlib import Path
from typing import Optional, List

import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
import shutil

app = FastAPI(title="SadTalker API", version="0.2.1")

# ---- 경로 기본값 (compose에서 마운트한다고 가정) ----
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/workspace/shared_data"))
INPUT_IMAGE_DIR = SHARED_DIR / "input_image"
APPLIO_OUT_DIR = SHARED_DIR / "applio_output_queue"
SADTALKER_OUT_DIR = SHARED_DIR / "sadtalker_output_queue"
WARMUP_DIR = SHARED_DIR / "warmup"

# SadTalker 스크립트 및 체크포인트 기본 경로
INFERENCE_PY = Path("/app/inference.py")
CHECKPOINT_DIR = Path("/app/checkpoints")

# 전역 batch_size 설정 파일(읽기 전용 정책)
BATCH_SIZE_FILE = SHARED_DIR / "batch_size.txt"

# 디렉토리 보장
for d in [SHARED_DIR, INPUT_IMAGE_DIR, APPLIO_OUT_DIR, SADTALKER_OUT_DIR, WARMUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---- 응답 모델 ----
class HealthResp(BaseModel):
    status: str = "ok"
    device: str = "unknown"
    inference_py: bool = True
    checkpoints: bool = True


@app.get("/health", response_model=HealthResp)
def health():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResp(
        status="ok",
        device=dev,
        inference_py=INFERENCE_PY.exists(),
        checkpoints=CHECKPOINT_DIR.exists(),
    )


def _pick_latest(directory: Path, patterns: List[str]) -> Optional[Path]:
    """디렉토리에서 패턴들 중 가장 최근 파일(수정시각 기준)을 반환"""
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(directory.glob(pat))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_batch_size_from_file() -> int:
    """
    {SHARED_DIR}/batch_size.txt에서 정수 읽기.
    없거나 잘못된 값이면 기본값 8 반환.
    """
    try:
        text = BATCH_SIZE_FILE.read_text(encoding="utf-8").strip()
        v = int(text)
        if v > 0:
            return v
    except Exception:
        pass
    return 8


def _ensure_warmup_assets() -> (Path, Path):
    """warmup 이미지/오디오가 없으면 생성 (간단 더미)"""
    img = None
    for ext in ["png", "jpg", "jpeg"]:
        cand = WARMUP_DIR / f"warmup_img.{ext}"
        if cand.exists():
            img = cand
            break
    if img is None:
        # 간단한 얼굴 스케치 이미지 생성
        try:
            from PIL import Image, ImageDraw
            img = WARMUP_DIR / "warmup_img.png"
            im = Image.new("RGB", (256, 256), (255, 255, 255))
            dr = ImageDraw.Draw(im)
            dr.ellipse((28, 28, 228, 228), outline=(0, 0, 0), width=3)  # face
            dr.ellipse((85, 95, 115, 125), fill=(0, 0, 0))              # left eye
            dr.ellipse((141, 95, 171, 125), fill=(0, 0, 0))             # right eye
            dr.arc((90, 130, 170, 190), start=10, end=170, fill=(0, 0, 0), width=3)  # smile
            im.save(img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create warmup image: {e}")

    wav = WARMUP_DIR / "warmup_out.wav"
    if not wav.exists():
        try:
            import numpy as np, soundfile as sf
            sr = 48000
            t = np.linspace(0, 0.30, int(sr * 0.30), endpoint=False)
            tone = (0.01 * np.sin(2 * np.pi * 440 * t)).astype("float32")
            sf.write(wav, tone, sr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create warmup wav: {e}")

    return img, wav


@app.post("/warmup")
def warmup(
    enhancer: str = Form("gfpgan"),
    preprocess: str = Form("crop"),
    size: int = Form(256),
    still: bool = Form(False),
):
    """
    실제 추론 1회 수행하여 모델/커널 예열.
    입력: warmup/warmup_img.(png|jpg|jpeg), warmup/warmup_out.wav (없으면 자동 생성)
    출력: warmup/warmup_sadtalker.mp4
    batch_size는 {SHARED_DIR}/batch_size.txt에서 읽기(없으면 8).
    """
    if not INFERENCE_PY.exists():
        raise HTTPException(status_code=500, detail=f"inference.py not found: {INFERENCE_PY}")

    img, wav = _ensure_warmup_assets()
    eff_bs = _read_batch_size_from_file()

    # 실행 전 기존 파일 스냅샷 (warmup 폴더 기준)
    before = {p: p.stat().st_mtime for p in WARMUP_DIR.glob("*.mp4")}

    cmd = [
        "python3", str(INFERENCE_PY),
        "--driven_audio", str(wav),
        "--source_image", str(img),
        "--result_dir", str(WARMUP_DIR),
        "--enhancer", enhancer,
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--preprocess", preprocess,
        "--size", str(size),
        "--batch_size", str(eff_bs),
    ]
    if still:
        cmd.append("--still")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            cwd="/app", env={**os.environ, "PYTHONPATH": "/app"},
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Warmup SadTalker failed (rc={e.returncode}). stdout[:600]={e.stdout[:600]} stderr[:600]={e.stderr[:600]}",
        )

    # 새로 생성된 파일 탐색
    new_file = None
    for _ in range(40):
        now_list = list(WARMUP_DIR.glob("*.mp4"))
        candidates = [p for p in now_list if p not in before or p.stat().st_mtime > before.get(p, 0) + 1]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            new_file = candidates[0]
            break
        time.sleep(0.5)

    if new_file is None:
        raise HTTPException(status_code=500, detail="Warmup: no mp4 was produced.")

    # warmup 파일명으로 정리
    final_path = WARMUP_DIR / "warmup_sadtalker.mp4"
    try:
        if new_file.resolve() != final_path.resolve():
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(new_file), str(final_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup move failed: {e}")

    return JSONResponse({
        "status": "ok",
        "message": "warmup done (real inference)",
        "image": str(img),
        "audio": str(wav),
        "output": str(final_path),
        "batch_size_effective": eff_bs,
        "cli": " ".join(cmd),
    })


@app.post("/infer")
def infer(
    input_image_path: Optional[str] = Form(None),
    input_audio_path: Optional[str] = Form(None),
    output_basename: Optional[str] = Form(None),
    enhancer: str = Form("gfpgan"),
    preprocess: str = Form("crop"),
    size: int = Form(256),
    still: bool = Form(False),
):
    if not INFERENCE_PY.exists():
        raise HTTPException(status_code=500, detail=f"inference.py not found: {INFERENCE_PY}")

    # 이미지/오디오 경로 결정 (생략 시 최신 자동 선택)
    if input_image_path:
        img = Path(input_image_path)
        if not img.exists():
            raise HTTPException(status_code=400, detail=f"input_image_path not found: {img}")
    else:
        img = _pick_latest(INPUT_IMAGE_DIR, ["*.png", "*.jpg", "*.jpeg"])
        if not img:
            raise HTTPException(status_code=400, detail="No image found in input_image/")

    if input_audio_path:
        aud = Path(input_audio_path)
        if not aud.exists():
            raise HTTPException(status_code=400, detail=f"input_audio_path not found: {aud}")
    else:
        aud = _pick_latest(APPLIO_OUT_DIR, ["*.wav", "*.mp3"])
        if not aud:
            raise HTTPException(status_code=400, detail="No audio found in applio_output_queue/")

    eff_bs = _read_batch_size_from_file()

    # 실행 전 스냅샷
    before = {p: p.stat().st_mtime for p in SADTALKER_OUT_DIR.glob("*.mp4")}

    cmd = [
        "python3", str(INFERENCE_PY),
        "--driven_audio", str(aud),
        "--source_image", str(img),
        "--result_dir", str(SADTALKER_OUT_DIR),
        "--enhancer", enhancer,
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--preprocess", preprocess,
        "--size", str(size),
        "--batch_size", str(eff_bs),
    ]
    if still:
        cmd.append("--still")

    try:
        subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            cwd="/app", env={**os.environ, "PYTHONPATH": "/app"},
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"SadTalker failed (rc={e.returncode}). stdout[:800]={e.stdout[:800]} stderr[:800]={e.stderr[:800]}",
        )

    # 이번 실행에서 새로 생긴 mp4들 모두 수집
    new_mp4s: List[Path] = []
    for _ in range(40):
        now_list = list(SADTALKER_OUT_DIR.glob("*.mp4"))
        new_mp4s = [p for p in now_list if p not in before or p.stat().st_mtime > before.get(p, 0) + 1]
        if new_mp4s:
            break
        time.sleep(0.5)

    if not new_mp4s:
        # fallback
        fallback = sorted(SADTALKER_OUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not fallback:
            raise HTTPException(status_code=500, detail="No mp4 was produced in output_queue.")
        new_mp4s = [fallback[0]]

    # 가장 최신(primary) 선택
    primary = max(new_mp4s, key=lambda p: p.stat().st_mtime)

    # 최종 파일명 결정
    if output_basename:
        final_name = f"{output_basename}.mp4"
    else:
        stem = Path(img).stem.replace("uploaded_image_", "")
        final_name = f"generated_video_{stem}.mp4"

    final_path = SADTALKER_OUT_DIR / final_name
    if final_path.exists():
        final_path = SADTALKER_OUT_DIR / f"{final_path.stem}_{int(time.time())}.mp4"

    # primary 이동(리네임)
    try:
        if primary.resolve() != final_path.resolve():
            shutil.move(str(primary), str(final_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to move result: {e}")

    # 나머지(extra) 정리(삭제)
    for p in new_mp4s:
        if p != final_path and p.exists():
            try:
                p.unlink()
            except Exception:
                # 삭제 실패는 치명적이지 않으니 무시
                pass

    return JSONResponse({
        "status": "ok",
        "message": "Video generated successfully by SadTalker",
        "image": str(img),
        "audio": str(aud),
        "output": str(final_path),
        "batch_size_effective": eff_bs,
        "cli": " ".join(cmd),
    })