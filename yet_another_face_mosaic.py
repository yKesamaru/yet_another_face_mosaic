import os
import cv2
import face_recognition
import PySimpleGUI as sg

# Initial setting
tolerance = 0.5
frame_drop = 0

vcap = cv2.VideoCapture('img/video.mp4', cv2.CAP_FFMPEG)

# Mosaic function
def mosaic(src, ratio):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

# Make window layout
sg.theme('Reddit')
layout = [
    [sg.Text('blur')],
    [sg.Image(key='display')],
    [sg.Button('terminate', key='terminate', button_color='red')]
]
window = sg.Window('blur', layout, location=(50, 50), disable_close=True)

# Encode face images in known_face directory
priset_face_image_encoding_list = []
for face in os.listdir('known_face'):
    pricet_face_img = face_recognition.load_image_file('known_face/' + face)
    priset_face_img_location = face_recognition.face_locations(pricet_face_img, 0, 'cnn')
    priset_face_image_encoding = face_recognition.face_encodings(pricet_face_img, priset_face_img_location, 10, 'small')
    priset_face_image_encoding_list.append(priset_face_image_encoding[0])

# Process each frame
frame_counter = 0
while True:
    ret, frame = vcap.read()
    if ret == False:
        break
    # Frame drop process
    if frame_drop >= frame_counter:
        frame_counter += 1
        continue
    frame_counter = 0

    dst = frame.copy()

    face_locations = face_recognition.face_locations(frame, 0, 'cnn')

    # Mosaicize only if a face is present
    if len(face_locations) > 0:
        # このフレームにknown_faceディレクトリ内の人物が存在する時、flagはTrue
        flag = False
        # 同じフレーム内のそれぞれの顔領域についてぼかし処理を行う
        for (top, right, bottom, left) in face_locations:
            face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)], 0, 'small')
            for priset_face_image_encoding in priset_face_image_encoding_list:
                distance = face_recognition.face_distance(priset_face_image_encoding, face_encoding)
                if distance[0] < tolerance:
                    flag = True
            if flag == False:
                # rectangle blur ======
                # dst[top:top + (bottom - top), left:left + (right - left)] = cv2.blur(dst[top:top + (bottom - top), left:left + (right - left)], (50, 50))
                # mosaic blur =========
                dst[top:top + (bottom - top), left:left + (right - left)] = mosaic(dst[top:top + (bottom - top), left:left + (right - left)], 0.1)
            flag = False
            

    event, _ = window.read(timeout=1)
    imgbytes = cv2.imencode(".png", dst)[1].tobytes()
    window["display"].update(data=imgbytes)
    if event == sg.WIN_CLOSED or event == 'terminate':
        break
vcap.release()
window.close()
