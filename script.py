import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import pytesseract
from pdf2image import convert_from_path
import math
import easyocr
import numpy as np
import re
import pdfplumber
import subprocess
from pathlib import Path


font_dir = "./font"


try:
    subprocess.run(["fc-cache", "-fv", font_dir], check=True)
    print(f"폰트 캐시 갱신 완료: {font_dir}")
except Exception as e:
    print(f"fc-cache 실행 실패: {e}")


model = YOLO("./model/co_epoch2.pt")
reader = easyocr.Reader(['en','ko'], model_storage_directory="./model",  download_enabled=False)


LABEL_MAP = {
    'Text': 'text',
    'Title': 'title',
    'Section-header': 'subtitle',
    'Formula': 'equation',
    'Table': 'table',
    'Picture': 'image'
}


def convert_to_images(input_path, temp_dir, dpi_pdf=741):
    """
    파일을 PIL 이미지 리스트로 변환
    - PDF: dpi로 렌더링
    - PPTX: PDF 변환 후 동일 처리
    - JPG/PNG: RGB 변환
    """
    ext = Path(input_path).suffix.lower()
    os.makedirs(temp_dir, exist_ok=True)

    if ext == ".pdf":
        images = convert_from_path(input_path, dpi=dpi_pdf, fmt="png")
        return [img.convert("RGB") for img in images]

    elif ext == ".pptx":
        subprocess.run([
            'soffice', "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path
        ], check=True)
        pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
        images = convert_from_path(pdf_path, dpi=dpi_pdf, fmt="png")
        return [img.convert("RGB") for img in images]

    elif ext in [".jpg", ".jpeg"]:
        img = Image.open(input_path).convert("RGB")
        clean_path = os.path.join(temp_dir, f"clean_{Path(input_path).stem}.png")
        img.save(clean_path, format="PNG", quality=100)
        return [Image.open(clean_path)]

    elif ext == ".png":
        img = Image.open(input_path).convert("RGB")
        return [img]


def scale_bbox_to_target(bbox, current_size, target_size):
    x1, y1, x2, y2 = bbox
    scale_x = target_size[0] / current_size[0]
    scale_y = target_size[1] / current_size[1]
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    ]

def clean_text(t):
    t = t.replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t
    
def extract_text(image_pil, bbox, lang='kor+eng', psm=1):
    x1, y1, x2, y2 = bbox
    cropped = image_pil.crop((x1, y1, x2, y2))
    config = f'--psm {psm}'
    text_raw = pytesseract.image_to_string(cropped, lang=lang, config=config)

    return clean_text(text_raw)


def scale_bbox_to_pdfminer(bbox, target_size, pdf_size):
    """
    YOLO 결과 bbox (target_size 기준) → pdfminer 좌표계(pdf_size) 변환
    """
    x1, y1, x2, y2 = bbox
    scale_x = pdf_size[0] / target_size[0]
    scale_y = pdf_size[1] / target_size[1]
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def safe_crop_bbox(bbox, page):
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page.bbox
    return (
        max(x0, px0),
        max(y0, py0),
        min(x1, px1),
        min(y1, py1),
    )


def extract_text_pdfplumber(pdf_path, bbox, target_size, page_number=0):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        pdf_w, pdf_h = page.width, page.height
        

        pdf_bbox = scale_bbox_to_pdfminer(bbox, target_size, (pdf_w, pdf_h))
        pdf_bbox = safe_crop_bbox(pdf_bbox, page)
        
        cropped = page.crop(pdf_bbox)
        return cropped.extract_text() or ""


def remove_special_chars(text):
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)


def apply_reading_order(predictions, y_tol=30):
    """
    리딩 오더 부여:
    - 중앙 x좌표로 좌/우 영역 분할
    - 각 영역 내에서는 y축 기준 정렬
    - 같은 줄(y_tol 이내)은 x축 기준으로 왼쪽→오른쪽
    - 최종 순서는 왼쪽 영역 먼저, 오른쪽 영역 나중
    """
    df = pd.DataFrame(predictions)
    coords = df['bbox'].apply(lambda b: list(map(int, str(b).replace(' ', '').split(','))))
    df['x1'] = coords.apply(lambda c: c[0])
    df['y1'] = coords.apply(lambda c: c[1])

    page_mid_x = (df['x1'].max() + df['x1'].min()) / 2

    def sort_within_region(region_df):
        """y축 + x축 기준으로 정렬"""

        region_df = region_df.sort_values(['y1', 'x1']).reset_index(drop=True)

        # 같은 줄 그룹핑
        orders = []
        current_order = 0
        prev_y = None
        for _, row in region_df.iterrows():
            if prev_y is None or abs(row['y1'] - prev_y) > y_tol:
                prev_y = row['y1']
            orders.append(current_order)
            current_order += 1
        region_df['order_tmp'] = orders
        return region_df

    left_df = df[df['x1'] <= page_mid_x].copy()
    right_df = df[df['x1'] > page_mid_x].copy()

    left_df = sort_within_region(left_df)
    right_df = sort_within_region(right_df)

    order_val = 0
    for df_region in [left_df, right_df]:
        for i in df_region.index:
            df_region.at[i, 'order'] = order_val
            order_val += 1

    df_ordered = pd.concat([left_df, right_df])
    df_ordered = df_ordered.sort_values('order').reset_index(drop=True)

    df_ordered = df_ordered.drop(columns=['x1', 'y1', 'order_tmp'], errors='ignore')
    return df_ordered.to_dict(orient='records')


def bbox_center_distance(bbox_str1, bbox_str2):
    x1a, y1a, x2a, y2a = map(int, bbox_str1.split(','))
    x1b, y1b, x2b, y2b = map(int, bbox_str2.split(','))
    cx1, cy1 = (x1a + x2a) / 2, (y1a + y2a) / 2
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)



def group_by_distance(preds, dist_thres=100):
    groups = []
    used = set()

    for i, p in enumerate(preds):
        if i in used:
            continue
        group = [p]
        used.add(i)
        for j in range(i+1, len(preds)):
            if j in used:
                continue
            if (p['ID'] == preds[j]['ID'] and
                p['category_type'] == preds[j]['category_type'] and
                bbox_center_distance(p['bbox'], preds[j]['bbox']) < dist_thres):
                group.append(preds[j])
                used.add(j)
        groups.append(group)
    return groups


def pptx_to_pdf(input_pptx, output_dir="./temp_pdf"):
    """
    PPTX 파일을 PDF로 변환
    LibreOffice(soffice) 필요
    """
    os.makedirs(output_dir, exist_ok=True)

    output_pdf = os.path.join(output_dir, Path(input_pptx).with_suffix(".pdf").name)

    try:
        subprocess.run([
            "soffice", "--headless",
            "--convert-to", "pdf",
            "--outdir", output_dir,
            input_pptx
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"PPTX → PDF 변환 실패: {input_pptx} → {e}")
        return None

    if not os.path.exists(output_pdf):
        print(f"PDF 파일 생성 안 됨: {output_pdf}")
        return None

    return output_pdf





def inference_one_image(id_val, image_pil, target_size,
                        conf_thres=0.15,
                        from_ppt=False, ppt_texts=None,
                        file_ext=None, pdf_path=None, page_number=0):

    if file_ext == ".pdf":
        conf_thres = 0.03  
    if file_ext == ".pptx":
        conf_thres = 0.03   

    resized_image = image_pil.resize((1024, 1024))
    temp_path = "_temp_image.png"
    resized_image.save(temp_path)

    results = model(source=temp_path, imgsz=1024, conf=0.05, verbose=False)[0]
    os.remove(temp_path)

    predictions = []

    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        label = results.names[int(cls)]
        if label not in LABEL_MAP:
            continue
        category_type = LABEL_MAP[label]
        

        if category_type in ['image', 'table']:
            if score < 0.09:
                continue
        else:
            if score < conf_thres:
                continue

        x1, y1, x2, y2 = scale_bbox_to_target(
            box.tolist(),
            (1024, 1024),
            target_size
        )

        text = ""

        if file_ext == ".pptx" and category_type in ['title','subtitle','text']:
            pdf_path = pptx_to_pdf(f"./data/test/{id_val}.pptx", "./temp_pdf")

            text = extract_text_pdfplumber(
                pdf_path, [x1, y1, x2, y2],
                target_size=target_size,
                page_number=page_number
            )
            if not text.strip(): 
                text = extract_text(image_pil, [x1, y1, x2, y2])

        elif file_ext == ".pdf" and category_type in ['title','subtitle','text']:
            text = extract_text_pdfplumber(
                pdf_path, [x1,y1,x2,y2],
                target_size=target_size,
                page_number=page_number
            )
            if not text.strip():
                text = extract_text(image_pil, [x1,y1,x2,y2])


        elif category_type in ['title','subtitle','text']:
            if file_ext in [".jpg", ".jpeg", "png"]:
                crop_img = image_pil.crop((x1, y1, x2, y2))
                crop_np = np.array(crop_img)
                ocr_results = reader.readtext(crop_np)
                text = " ".join([txt for _, txt, conf in ocr_results if conf > 0.01])
            else:
                text = extract_text(image_pil, [x1, y1, x2, y2])



        predictions.append({
            'ID': id_val,
            'category_type': category_type,
            'confidence_score': score.cpu().item(),
            'order': None,
            'text': text,
            'bbox': f'{x1}, {y1}, {x2}, {y2}'
        })

    predictions = apply_reading_order(predictions, y_tol=15)
    return predictions




def bbox_gap_both(box1, box2):
    """
    두 박스의 좌상단(x1,y1), 우하단(x2,y2) 거리 계산
    """
    x1a, y1a, x2a, y2a = map(int, str(box1).replace(" ", "").split(","))
    x1b, y1b, x2b, y2b = map(int, str(box2).replace(" ", "").split(","))

    gap_topleft = math.sqrt((x1a - x1b) ** 2 + (y1a - y1b) ** 2)
    gap_bottomright = math.sqrt((x2a - x2b) ** 2 + (y2a - y2b) ** 2)

    return gap_topleft, gap_bottomright


def inference(test_csv_path="./data/test.csv", output_csv_path="./output/submission.csv"):
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    temp_image_dir = "./temp_images"
    os.makedirs(temp_image_dir, exist_ok=True)

    csv_dir = os.path.dirname(test_csv_path)
    test_df = pd.read_csv(test_csv_path)
    all_preds = []

    for _, row in test_df.iterrows():
        id_val = row['ID']
        raw_path = row['path']
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))
        target_width = int(row['width'])
        target_height = int(row['height'])

        if not os.path.exists(file_path):
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        try:
            ext = Path(file_path).suffix.lower()
            images = convert_to_images(file_path, temp_image_dir)

            for i, image in enumerate(images):
                full_id = f"{id_val}_p{i+1}" if len(images) > 1 else id_val
                preds = inference_one_image(
                    full_id, image, (target_width, target_height),
                    from_ppt=(ext == ".pptx"), ppt_texts=None,
                    file_ext=ext, pdf_path=file_path, page_number=i
                )
                all_preds.extend(preds)

            print(f"예측 완료: {file_path}")

        except Exception as e:
            print(f"❌ 처리 실패: {file_path} → {e}")

    grouped_preds = []
    for group in group_by_distance(all_preds, dist_thres=40):
        grouped_preds.extend(group)

    test_df = pd.read_csv("./data/test.csv")

    result_df = pd.DataFrame(grouped_preds)
    result_df.to_csv(output_csv_path, index=False, encoding='UTF-8-sig')
    print(f"저장 완료: {output_csv_path}")


if __name__ == "__main__":
    inference("./data/test.csv", "./output/submission.csv")


