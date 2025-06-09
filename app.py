from flask import Flask, jsonify
import pymysql
import torch
import os
import requests
import gdown
import zipfile
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ 경로 설정
zip_path = "clip_model.zip"
extract_path = "clip_finetuned_model"

# ✅ 모델 폴더가 없으면 처음 실행 시 다운로드 + 압축 해제
if not os.path.exists(extract_path):
    print("📦 모델 다운로드 중...")
    gdown.download(id="1ubujA5oDq_-wg0W-r6BmJi6NM0cIvFS4", output=zip_path, quiet=False)


    print("📦 압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ 모델 준비 완료")

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ CLIP 모델 로드
model = CLIPModel.from_pretrained(extract_path).to(device)
processor = CLIPProcessor.from_pretrained(extract_path)
model.eval()

# ✅ 클래스 태그
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

# ✅ DB 설정
DB_CONFIG = {
    'host': 'project-db-cgi.smhrd.com',
    'port': 3307,
    'user': 'cgi_24K_AI4_p3_2',
    'password': 'smhrd2',
    'db': 'cgi_24K_AI4_p3_2',
    'charset': 'utf8mb4'
}

# ✅ 현재 실행 환경 (환경변수 없으면 기본값은 'development')
MODE = os.environ.get("FLASK_MODE", "development")
BASE_NODE_URL = "http://localhost:5000" if MODE == "development" else "https://tripd.onrender.com"

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
        image_url = f"{BASE_NODE_URL}/uploads/{filename}"  # ✅ 로컬/배포 자동 분기

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
    app.run(host="0.0.0.0", port=6006)
