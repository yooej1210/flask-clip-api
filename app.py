from flask import Flask, jsonify
import pymysql
import torch
import os
import requests  # ✅ 추가
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# ✅ 디바이스 설정 (GPU가 있으면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Fine-tuned CLIP 모델 로드
model = CLIPModel.from_pretrained("./clip_finetuned_model").to(device)
processor = CLIPProcessor.from_pretrained("./clip_finetuned_model")
model.eval()

# ✅ 분류 대상 태그 클래스 (수정 가능)
class_names = ["food", "people", "landscape", "accommodation"]

# ✅ 이미지 분류 함수 (image_path → image_url로 변경)
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
    'password': 'smhrd2',  # ← 본인 비밀번호로 변경
    'db': 'cgi_24K_AI4_p3_2',         # ← 본인 DB 이름으로 변경
    'charset': 'utf8mb4'
}

# ✅ VSCode 서버 쪽 uploads 폴더 절대 경로 (이제 사용 안함)
UPLOADS_DIR = "C:/Users/smhrd/Desktop/pokachip/server/uploads"

# ✅ Flask API 엔드포인트: /classify
@app.route('/classify', methods=['POST'])
def classify_images():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # tags가 비어있는 항목들만 선택
    cursor.execute("SELECT photo_idx, file_name FROM photo_info WHERE tags IS NULL OR tags = ''")
    photos = cursor.fetchall()

    classified_count = 0

    for photo in photos:
        photo_idx = photo['photo_idx']
        # 슬래시 정규화 + 파일명만 추출
        filename = os.path.basename(photo['file_name'].replace('\\', '/'))

        # ✅ 이미지 URL 경로로 변경 (Node.js 서버 주소로 바꿔야 함)
        image_url = f"https://your-node-app.onrender.com/uploads/{filename}"

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

# ✅ 서버 실행
if __name__ == '__main__':
    app.run(port=6006)
