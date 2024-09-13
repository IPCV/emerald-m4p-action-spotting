# EMERALD Action Spotting

Overview
This test scenario involves two phases: the ffmpeg splitter and the video processing. The ffmpeg splitter breaks down a video file into chunks, which are then processed by a video receiver and sender service.

How the Receiver and Sender Work
Receiver: The receiver service listens for incoming chunk uploads and stores them on the disk in the uploads folder. Upon receiving a chunk, it sends a queue message containing the file name for the processor to read. The processor.py script can run in parallel using 4 threads to efficiently handle multiple chunks simultaneously.

Sender: The sender service continuously sends video chunks to the receiver and checks the processing status of each chunk. It ensures that chunks are sent again if they are still being processed until they are successfully handled.

### Running the Test Scenario

1. **Insert your video into the `media` folder.**

2. **Create the chunks:**

   
   Navigate to the `ffmpeg` folder and run the Docker Compose setup:

   Note: Change the VIDEO_INPUT_FILE in the ffmpeg Docker Compose file to match your video file name

   ```sh
   cd ffmpeg
   docker-compose up --build
   ```

   This will create video chunks in the `output-chunks` folder.

3. **Start the receiver and sender services:**

   Navigate to the `src` folder and run the Docker Compose setup:

   ```sh
   cd ../src
   docker-compose up --build
   ```

   This will start the receiver and sender services.
   For testing purposes, you should remove the uploads volume in the Docker Compose configuration. This will ensure that the uploads folder is deleted when you stop the container, allowing for a clean slate on each run.
   Note: if you want to run just one of the services, you can run 
   ```sh
   docker-compose up video-receiver|simple-sender
   ```

### AI Video Processing Development

For AI video processing development, you should edit the `processor.py` file located in the `src/receiver` directory. This script is responsible for processing the chunks received by the video-receiver. Ensure any new dependencies are added to the respective `requirements.txt` files.

## Acknowledgements

This project is a collaboration effort between MOG Technologies and the Universitat Pompeu Fabra. The authors acknowledge support by MICINN/FEDER UE project, ref. PID2021-127643NB-I00, and the support of the European Commission, Horizon Europe Programme, EMERALD Project 101119800.

<p align="center" float="left">
  <img src="assets/ministerio_logo.png" alt="Ministerio" height="100"/>&nbsp;&nbsp;
  <img src="assets/emerald_logo.png" alt="EMERALD" height="100"/>
</p>
