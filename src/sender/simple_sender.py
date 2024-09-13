import os
import time
import requests
import threading
import queue
from utils.logs import setup_logger

logger = setup_logger("sender")

CHUNKS_FOLDER = '/chunks'
URL = os.getenv('RECEIVER_URL')
CHUNK_INTERVAL = 2 # seconds

id_queue = queue.PriorityQueue()

def send_chunk(file_path):
    logger.info(f'Going for {file_path}')
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
        upload_url= f'{URL}/upload'
        logger.info(upload_url)
        response = requests.post(upload_url, files=files)
        logger.info(f"Sent {file_path}, response: {response.json()}, status: {response.status_code}")
        if response.status_code == 200: 
            response_data = response.json()
            chunk_id = response_data.get('id')
            if chunk_id:
                # Add ID to the queue with the lowest priority | same priotiry then read the priority 
                id_queue.put((0, chunk_id))

def get_status(id):
    response = requests.get(f"{URL}/{id}/status")
    return response

def manage_chunks():
    logger = setup_logger("processor")
    while True:
        priority, chunk_id = id_queue.get()
        response = get_status(chunk_id)
        logger.info(response)
        status_code = response.status_code
        response_data = response.json()
        
        if status_code == 200 and response_data.get('status') == 'processed':
            response = requests.get(f"{URL}/{chunk_id}/metadata")
            metadata_status_code = response.status_code
            if metadata_status_code == 200: 
                logger.info(f"Chunk ID {chunk_id} processing completed. Deleting ID.")
                logger.debug(f"Metadata for {chunk_id} - {response.json()}")
                requests.delete(f"{URL}/videos/{chunk_id}")
                
            else: 
                # should not be necessary
                logger.debug(f"Error finding chunk {chunk_id} metadata. Re-queuing with increased priority.")
                id_queue.put((priority + 1, chunk_id)) 
                logger.debug(f"Re-added {chunk_id}")

        else:
            logger.info(f"Chunk ID {chunk_id} still processing. Re-queuing with increased priority.")
            id_queue.put((priority + 1, chunk_id)) 
            # could a max number of requests for this to  not get stuck from any type of mistake
            # after max attempt reintroduce to queue with minimum priority
        
        id_queue.task_done()
        time.sleep(1) 

def main():
    logger.info("main")
    logger.info(os.listdir(CHUNKS_FOLDER))
    chunks = sorted([os.path.join(CHUNKS_FOLDER, f) for f in os.listdir(CHUNKS_FOLDER) if f.endswith('.mp4')])
    logger.info(chunks)
    for chunk in chunks:
        send_chunk(chunk)
        time.sleep(CHUNK_INTERVAL)

if __name__ == '__main__':
    # Start the manage_chunks function in a separate thread
    manage_thread = threading.Thread(target=manage_chunks, daemon=True)
    manage_thread.start()
    
    main()
    
    # Wait for the main function to complete all chunk uploads
    manage_thread.join()
