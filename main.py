import os
import cv2
import numpy as np
import ls

# 設定ファイル
prototxt = "res/prototxt/deploy.prototxt"
model = "res/model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
image_dir_path = "res/test_images"
result_path = "result"

# 検出した部位の信頼度の下限値
confidence_limit = 0.5
# 設定ファイルからモデルを読み込み
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# resultフォルダを作成（exist_ok: 既に作成したいフォルダが存在していたら無視する）
os.makedirs(result_path, exist_ok=True)

for image_name in ls.ls_file(image_dir_path):

    # 解析対象の画像を読み込み
    image = cv2.imread(os.path.join(image_dir_path, image_name))

    # 300x300に画像をリサイズ、画素値を調整
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    # 顔検出の実行
    net.setInput(blob)
    detections = net.forward()
    # 検出部位をチェック
    for i in range(0, detections.shape[2]):
        # 信頼度
        confidence = detections[0, 0, i, 2]
        # 信頼度が下限値以下なら無視
        if confidence < confidence_limit:
            continue
        # 検出結果を描画
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 6)
        cv2.putText(
            image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 10
        )

    print(os.path.splitext(image_name)[0])
    cv2.imwrite(os.path.join(result_path, image_name), image)
