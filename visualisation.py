import cv2

SEABORN_RED = (82, 78, 196)
SEABORN_GREEN = (104, 168, 85)
SEABORN_BLUE = (176, 114, 76)


def process_img_for_angle_visualization(img, angle, pred_angle, frame):
    font = cv2.FONT_HERSHEY_COMPLEX

    img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)

    h, w = img.shape[0:2]

    # add black rectangle behind text for readability
    img = cv2.rectangle(img, (0, 0), (455, 95), (0, 0, 0), -1)

    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(30, 20),
                fontFace=font, fontScale=0.8, color=SEABORN_BLUE, thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(30, 50), fontFace=font,
                fontScale=0.8, color=SEABORN_GREEN, thickness=1)
    cv2.putText(img, 'predicted angle: ' + str(pred_angle), org=(30, 80),
                fontFace=font, fontScale=0.8, color=SEABORN_RED, thickness=1)

    # apply a line representing the steering angle
    cv2.line(img, (int(w/2), int(h)), (int(w/2-angle*w/4), int(h/2)),
             SEABORN_GREEN, thickness=5)

    if pred_angle is not None:
        cv2.line(img, (int(w/2), int(h)), (int(w/2-pred_angle*w/4), int(h/2)),
                 SEABORN_RED, thickness=3)
    return img
