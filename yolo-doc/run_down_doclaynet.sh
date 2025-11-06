#!/bin/bash

# 1. DocLayNet 데이터셋 다운로드
curl -L -o DocLayNet_core.zip https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip

# 2. datasets 디렉토리 생성
mkdir -p datasets_doc

# 3. 다운로드한 zip 파일 이동
mv DocLayNet_core.zip datasets_doc/

# 4. 압축 해제 후 zip 파일 삭제
cd datasets_doc/ && unzip DocLayNet_core.zip && rm DocLayNet_core.zip
cd ..

# 5. 변환 스크립트 실행
python convert_dataset_doc.py


chmod +x run_doclaynet.sh   
./run_doclaynet.sh       

