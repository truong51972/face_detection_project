import torch

import numpy as np
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

def extract(num_of_imgs: int = 300, draw: bool = True):
    '''
    Return a list of feature.
    '''

    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(thresholds= [0.8, 0.9, 0.9], min_face_size= 300, margin = 50, keep_all=False, post_process=False, device = device)

    count = num_of_imgs
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    features = []
    if num_of_imgs >= 0:
        print('\nReady to record!')
        input('Press enter to record!')

    while cap.isOpened():
        isSuccess, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if isSuccess:
            boxes, probs, points = mtcnn.detect(frame_rgb, landmarks=True)
            if boxes is not None:
                for i, (box, prob, point) in enumerate(zip(boxes, probs, points)):
                    img_tensor = extract_face(frame, box)

                    batch_imgs = img_tensor.unsqueeze(0).to(device)

                    embedded_tensor = resnet(batch_imgs)

                    embedded_list = embedded_tensor[0].tolist()
                    features.append(embedded_list)
                    
                    count -= 1

                    if draw:
                        bbox = list(map(int,box.tolist()))
                        cv2.putText(img=frame,
                                    text=str(round(prob,2)),
                                    org=(bbox[0] - 10,bbox[1] - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(0,0,255),
                                    thickness=3)
                        
                        cv2.putText(img=frame,
                                    text=str(count),
                                    org=(0, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(0,0,255),
                                    thickness=3)
                        
                        frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)

                        for p in point:
                            p = list(map(int,p.tolist()))
                            cv2.circle(frame, p, 5, [0, 0, 255], -1)

        cv2.imshow('Face Detection', frame)

        if count == 0: break
        
        if cv2.waitKey(1)&0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Recording completed!')
    return features

if __name__ == "__main__":
    extract()