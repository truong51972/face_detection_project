import cv2
import torch

from packages.face_classify import Classifier_model
from packages.query import Database

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face


def detect():
    model = Classifier_model(path='./model/svm_model.pkl')

    database = Database()

    idx_to_name = database.get_idx_to_name()

    database.close()



    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(thresholds= [0.7, 0.8, 0.8], min_face_size= 100, margin = 20, keep_all=False, post_process=False, device = device)    


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
                    img_tensor = extract_face(frame, box)

                    batch_imgs = img_tensor.unsqueeze(0).to(device)
                    embedded_tensor = resnet(batch_imgs)
                    embedded_list = embedded_tensor.tolist()

                    classifier_prob, classifier_idx = model.predict(feature=embedded_list)
                    name = idx_to_name[classifier_idx]

                    if True:
                        bbox = list(map(int,box.tolist()))
                        cv2.putText(img=frame,
                                    text= f"{name}-{round(classifier_prob*prob, 2)}",
                                    org=(bbox[0] - 10,bbox[1] - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.7,
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