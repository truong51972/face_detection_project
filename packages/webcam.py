import cv2
from facenet_pytorch import MTCNN
import torch

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = MTCNN(thresholds= [0.7, 0.8, 0.8] ,keep_all=False, device = device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while cap.isOpened():
    isSuccess, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if isSuccess:
        boxes, probs, points = mtcnn.detect(frame_rgb, landmarks=True)
        if boxes is not None:
            for i, (box, prob, point) in enumerate(zip(boxes, probs, points)):
                bbox = list(map(int,box.tolist()))

                cv2.putText(img=frame,
                            text=str(round(prob,2)),
                            org=(bbox[0] - 10,bbox[1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0,0,255),
                            thickness=3)
                frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)

                for p in point:
                    p = list(map(int,p.tolist()))
                    cv2.circle(frame, p, 5, [0, 0, 255], -1)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
