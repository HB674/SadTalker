#!/usr/bin/env bash
set -e

# === 디렉터리 생성(있으면 통과) ===
mkdir -p ./checkpoints
mkdir -p ./gfpgan/weights

# === 편의 함수: 없을 때만 다운로드 ===
dl() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "↪️  Skip (exists): $out"
  else
    echo "⬇️  Download: $url -> $out"
    wget -L --no-verbose -O "$out" "$url"
  fi
}

# ===== SadTalker (new links) =====
dl "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar" \
   "./checkpoints/mapping_00109-model.pth.tar"

dl "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar" \
   "./checkpoints/mapping_00229-model.pth.tar"

dl "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" \
   "./checkpoints/SadTalker_V0.0.2_256.safetensors"

dl "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors" \
   "./checkpoints/SadTalker_V0.0.2_512.safetensors"

# ===== Enhancer (GFPGAN / facexlib) =====
dl "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth" \
   "./gfpgan/weights/alignment_WFLW_4HG.pth"

dl "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
   "./gfpgan/weights/detection_Resnet50_Final.pth"

dl "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
   "./gfpgan/weights/GFPGANv1.4.pth"

dl "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" \
   "./gfpgan/weights/parsing_parsenet.pth"

echo "✅ download_models.sh done."
