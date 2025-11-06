## Visually-rich Document Understanding 
<img width="1055" height="214" alt="image" src="https://github.com/user-attachments/assets/0e6007cf-3473-4c5a-8435-f8606295919b" />  
https://dacon.io/competitions/official/236564/overview/description  

<p align="center">
  <img src="TEST.png" alt="YOLO13x_doclaynet 결과" width="300"/>
</p>

# Solution


- If you want using my weights.pt. Please contact to us( 8536048@gmail.com )
- Then, we will give you our models.zip

## `<yolo-doclaynet>` Folder

1. Clone the **YOLOv13** repository and use the **ultralytics** package from it.
2. Download the **DocLayNet 30k** dataset.
3. Pre-train `yolov13x.pt` (from the YOLOv13 GitHub repository) on the DocLayNet dataset.
4. Fine-tune the model according to the slides/guidelines.
5. Place the final trained `.pt` model into the `./model` folder.

---
## User Guideline

1. Clone the **EasyOCR** repository.
2. Clone the **Yolo13v** repository and get ultralytics folder copy and paste.
3. Download the three EasyOCR pretrained weights and place them inside the `model` folder.
   - **Note:** No additional fine-tuning was performed, and the results were actually better without fine-tuning.
   - You can get here -> https://huggingface.co/felflare/EasyOCR-weights/tree/main here.
   - For using our weights, Please contact us. (qmdlghfl2@gmail.com)
4. Run `script.py`.
5. Use `check_val_visual_score.ipynb` (evaluation code notebook) to visualize bounding boxes.
6. After monitoring the results, execute `make_zip.ipynb` to generate the final submission zip file.

## NOTE 

1. This is not full repository.(Because easyocr, ultralytics, etc .. are empty folders. Just architecture)
2. If you want use this code, then follow **the User Guideline**.

## Reference 

- YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception (Mengqi Lei et. al, 2025 CVPR)
- https://github.com/iMoonLab/yolov13
- https://github.com/JaidedAI/EasyOCR
