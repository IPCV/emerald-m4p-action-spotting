import cv2
import json
import os
import time
import torch
from utils.logs import setup_logger

from transforms import transform_image

logger = setup_logger("Processor")

actions = {
  "Background",
  "Penalty",
  "Kick-off",
  "Goal",
  "Substitution",
  "Offside",
  "Shots on target",
  "Shots off target",
  "Clearance",
  "Ball out of play",
  "Throw-in",
  "Foul",
  "Indirect free-kick",
  "Direct free-kick",
  "Corner",
  "Yellow card",
  "Red card",
  "Yellow->red card"
}

def save_metadata_as_json(metadata, file_path):
    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def extract_frames(video_path, sample_rate=2.):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        logger.error("Error: Could not open video.")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    interval = int(fps / sample_rate)
    frame_count = 0

    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_count += 1

    video_capture.release()
    return frames


def process_chunk(args):
    video_path, model, device = args
    time0 = time.time()

    frames = extract_frames(video_path)
    device = torch.device(device)

    batch = [transform_image(f).unsqueeze(0).to(device) for f in frames]
    batch = torch.cat(batch, dim=0).squeeze()

    prediction = model(batch)
    prediction = prediction.detach().cpu().numpy()[0].tolist()

    logger.debug(json.dumps(prediction))

    time1 = time.time()
    
    predictions = {} 
    for i, pred in enumerate(actions):
        predictions[pred] = prediction[i]

    logger.debug(f"Metadata exported: {predictions}")

    metadata = {
        "status": "processed",
        "video_path": video_path,
        "prediction": predictions
    }

    metadata_file_path = os.path.join(os.path.split(video_path)[0], 'metadata.json')

    logger.debug(f"Saving metadata file to {metadata_file_path}")

    save_metadata_as_json(metadata, metadata_file_path)

    time2 = time.time()

    logger.debug(f'Total time : {time2 - time0} - Process: {time1 - time0} - Metadata: {time2 - time1}')
