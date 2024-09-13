## ingest video into chunks and sends it via multipart uploadimport requests
import requests
import os 
from ffmpeg import FFmpeg, Progress


url = "http://localhost:5000/upload"
main_file_path = "../media/esquerda10s0805.mp4"
segments_dir = 'segments_esq10s'

## either function and save or send in real-time
def split_into_chunks(main_file,segments_dir):
    
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(
            "rtsp://username:password@127.0.0.1/cam",
            rtsp_transport="tcp",
            rtsp_flags="prefer_tcp",
        )
        .output("output.mp4", vcodec="copy")
    )
    # try on progress send to the post 
    # by time or frames prcoesse 
    @ffmpeg.on("progress")
    def time_to_terminate(progress: Progress):
        init = 0
        chunk_size_frames = 50  
        # 25 fps
        if progress.fps > init + chunk_size_frames :
            #go to files
            init += chunk_size_frames
            
    ffmpeg.execute()


try:
    for name in os.listdir(segments_dir):
        # Open file
        with open(os.path.join(segments_dir, name)) as f:
            r = requests.post(url, files={'file': f})
except FileNotFoundError:
    print("The file was not found.")
except requests.exceptions.RequestException as e:
    print("There was an exception that occurred while handling your request.", e)