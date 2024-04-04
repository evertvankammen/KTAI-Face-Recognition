import cv2
import mmcv
import numpy as np
import os
import torch
from IPython import display
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
videoPath = os.path.join("data", "pictures", "Friends.mp4")
mtcnn = MTCNN(keep_all=True, device=device)

video = mmcv.VideoReader(videoPath)
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

display.Video(videoPath, width=640)

frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    # Controleer of `boxes` niet None is voordat je probeert er doorheen te itereren
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
            # Voeg toe aan frame lijst
            frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    else:
        # Behandel het geval waarin geen objecten worden gedetecteerd
        print("Geen objecten gedetecteerd.")

d = display.display(frames_tracked[0], display_id=True)
i = 1
try:
    while True:
        d.update(frames_tracked[i % len(frames_tracked)])
        i += 1
except KeyboardInterrupt:
    pass

dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()