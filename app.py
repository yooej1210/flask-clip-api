from flask import Flask, jsonify
import pymysql
import torch
import os
import requests
import gdown
import zipfile
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# ✅ 경로 설정
zip_path = "clip_model.zip"
extract_path = "clip_finetuned_model"

# ✅ 모델 폴더가 없으면 처음 실행 시 다운로드 + 압축 해제
# ✅ 모델 폴더가 없으면 다운로드 + 압축 해제
if not os.path.exists(extract_path):
    print("📦 모델 다운로드 중...")
    gdown.download(id="1P8ynqxG221Qg81_KT4upfieA-knEeGf3", output=zip_path, quiet=False)

    print("📦 압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ 모델 준비 완료")

# ✅ 디바이스 설정 (GPU가 있으면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Fine-tuned CLIP 모델 로드
model = CLIPModel.from_pretrained(extract_path).to(device)
processor = CLIPProcessor.from_pretrained(extract_path)
model.eval()

# ✅ 분류 대상 태그 클래스 (수정 가능)
class_names = ["food", "people", "landscape", "accommodation"]

# ✅ 이미지 분류 함수
def predict_tag(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    inputs = processor(text=class_names, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        pred_index = torch.argmax(probs).item()
        return class_names[pred_index]

# ✅ MySQL DB 접속 설정 (포트 3307)
DB_CONFIG = {
    'host': 'project-db-cgi.smhrd.com',
    'port': 3307,
    'user': 'cgi_24K_AI4_p3_2',
    'password': 'smhrd2',
    'db': 'cgi_24K_AI4_p3_2',
    'charset': 'utf8mb4'
}

@app.route('/classify', methods=['POST'])
def classify_images():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT photo_idx, file_name FROM photo_info WHERE tags IS NULL OR tags = ''")
    photos = cursor.fetchall()

    classified_count = 0

    for photo in photos:
        photo_idx = photo['photo_idx']
        filename = os.path.basename(photo['file_name'].replace('\\', '/'))
        image_url = f"https://your-node-app.onrender.com/uploads/{filename}"  # ← 실제 URL로 바꿔야 함

        try:
            tag = predict_tag(image_url)
            cursor.execute("UPDATE photo_info SET tags = %s WHERE photo_idx = %s", (tag, photo_idx))
            conn.commit()
            print(f"✅ 분류 완료: {filename} → {tag}")
            classified_count += 1
        except Exception as e:
            print(f"⚠️ 예측 실패 ({filename}): {e}")

    cursor.close()
    conn.close()

    return jsonify({
        "status": "success",
        "classified": classified_count
    })

if __name__ == '__main__':
    app.run(port=6006)
