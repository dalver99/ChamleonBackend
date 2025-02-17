import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import base64
import time
import requests
from PIL import Image
from datetime import datetime
import pymysql
from pymysql.cursors import DictCursor
from mysql.connector import Error
import cv2
import numpy as np
import torch
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from helpers import visualize_all, get_parser_zip, upload_gaze
import json


app = FastAPI()

with open(f'./SQL_parameter.json', 'r') as file:
    par = json.load(file)
    
# Database connection function
def get_db_connection():
    connection = pymysql.connect(
        host=par['host'],
        user=par['username'],
        password=par['password'],
        database=par['database'],
        port=par['port'],
        cursorclass=DictCursor  # 쿼리 결과를 딕셔너리 형태로 반환
    )
    return connection

    
# DB 연결 테스트용 엔드포인트
async def test_db():
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT * FROM frame LIMIT 5")
        result = cursor.fetchall()
        cursor.close()
        return {"success": True, "data": result}
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": str(e)},
            status_code=500
        )
        
# 이미지 img로 업로드
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"Failed to open image: {str(e)}"}, status_code=400)

    raw_file_name = file.filename

    captured_at = datetime.now()
    imgur_url = upload_to_imgur(img)
    
    if imgur_url:
        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            insert_img = "INSERT INTO frame(ad_id, camera_id, image_path, captured_at, name) VALUES (96, 1, %s, %s, %s)"
            cursor.execute(insert_img, (imgur_url, captured_at, raw_file_name))
            connection.commit()
            cursor.close()
            connection.close()
            print(imgur_url, captured_at, raw_file_name)
            print('DB 연결완료')
        except Exception as e:
            print(f"DB 저장 실패: {e}")
            return JSONResponse(content={"success": False, "message": f"Failed to save to DB: {str(e)}"}, status_code=500)
        
        return {
            "success": True, 
            "imgur_url": imgur_url,
            "filename": raw_file_name,
            "captured_at": captured_at.isoformat()
        }
    else:
        return JSONResponse(content={"success": False, "message": "Failed to upload image"}, status_code=500)

        
def upload_to_imgur(img, retry_count = 3):
    with open(f'./imgur_api.json', 'r') as file:
        client = json.load(file)
    IMGUR_CLIENT_ID = client['client_id'] # Imgur에서 개인 API 발급 후 사용가능
        
    url = "https://api.imgur.com/3/image"
    headers = {
        "Authorization": f"Client-ID {IMGUR_CLIENT_ID}",
        "User-Agent": "My Python App"
    }

    buffered = io.BytesIO()
    img_rgb = img.convert('RGB')
    img_rgb.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    encoded_string = base64.b64encode(img_byte).decode('utf-8')

    data = {"image": encoded_string}

    response = requests.post(url, headers=headers, data=data, verify=False)

    if response.status_code == 200:
        print(f'Success! Image uploaded to: {response.json()["data"]["link"]}')
        return response.json()["data"]["link"]
    elif retry_count > 0:
        print('Upload failed. Retrying...')
        time.sleep(5)
        return upload_to_imgur(img, retry_count - 1)
    else:
        print(f'Failed to upload image: {response.content}')
        return None        

@app.post("/process_image/")
async def process_image(x1: float, y1: float, x2: float, y2: float, file: UploadFile):
    setup_default_logging()

    # 이미지 읽기
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()  # 이미지 파일 확인
        img = Image.open(io.BytesIO(contents))  # 다시 열기
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to open image: {str(e)}"}, status_code=400)

    # 이미지 imgur & db 업로드
    await file.seek(0)  # 파일 포인터를 처음으로 되돌립니다
    upload_result = await upload_image(file)
    if not upload_result["success"]:
        return JSONResponse(content={"error": "Failed to upload image"}, status_code=500)
    
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    width, height = img.size

    # ROI 설정
    roi_state = {'box_points': [(x1, y1), (x2, y2)], 'roi_selected': True}  #x1,y1: 좌하단, x2,y2: 우상단

    # cuda 사용 안됨
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MiVOLO 예측
    print("Loading MiVOLO Model...")
    args = get_parser_zip().parse_args([])
    predictor = Predictor(args)
    detected_objects, _ = predictor.recognize(img_cv2)
    print("Predicting with YOLO & MiVOLO Model...")

    # YOLO 결과 처리
    yolo_results = detected_objects.yolo_results
    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    classes = yolo_results.boxes.cls.cpu().numpy()

    face_class_indices = [0]  # 0: 'person'
    face_indices = np.where(np.isin(classes, face_class_indices))[0]

    if len(face_indices) == 0:
        print('No bounding boxes detected by YOLO. Exiting.')
        return
    
    bboxes = boxes[face_indices]
    ages = [detected_objects.ages[i] for i in face_indices]
    genders = [detected_objects.genders[i] for i in face_indices]

    # Gazelle 모델 로딩 및 예측
    print("Loading Gazelle Model...")
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14_inout', source='github')
    model.eval()
    model.to(device)

    img_tensor = transform(img).unsqueeze(0).to(device)
    norm_bboxes = [[bbox / np.array([width, height, width, height]) for bbox in bboxes]]
    gazelle_input = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }

    with torch.no_grad():
        output = model(gazelle_input)

    # 결과 시각화
    inout_scores = output['inout'][0] if output['inout'] is not None else None
    raw_file_name = file.filename

    overlay_image = visualize_all(
        roi_state['box_points'],
        img,
        output['heatmap'][0],
        norm_bboxes[0],
        raw_file_name,
        inout_scores,
        ages,
        genders,
        inout_thresh=0.5,
    )
    
    imgur_link = upload_to_imgur(overlay_image)
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        print('db 업데이트 진행중')
        # 🔹 id 가져오기 (SQL 실행)
        cursor.execute("SELECT id FROM frame WHERE image_path = %s", (upload_result["imgur_url"],))
        frame_row = cursor.fetchone()
        if frame_row is None:
            return JSONResponse(content={"success": False, "message": f"No matching id found for image: {upload_result['imgur_url']}"}, status_code=404)

        frame_id = frame_row["id"]  # id 값 추출 (딕셔너리 형태이므로 키로 접근)
        
        insert_img = "UPDATE frame SET predicted_image_path=%s WHERE name = %s"
        cursor.execute(insert_img, (imgur_link, raw_file_name))
        connection.commit()  # 변경사항을 커밋
        cursor.close()
        connection.close()  # 연결 종료
        return {"success": True, "frame_id": frame_id, "message": "Image processed and database updated successfully"}
    except Exception as e:
        connection.rollback()  # 오류 발생 시 롤백
        cursor.close()
        connection.close()  # 연결 종료
        return JSONResponse(
            content={"success": False, "message": str(e)},
            status_code=500
        )  


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)