FROM alpine:latest

# Install ffmpeg
RUN apk add --no-cache ffmpeg

# Define video path and segment time 
ENV VIDEO_INPUT_FILE esquerda10s0805.mp4 
ENV SEGMENT_TIME 2

# Create a directory for the output files
RUN mkdir -p /output

# Define the command to run ffmpeg
CMD ffmpeg -i media/$VIDEO_INPUT_FILE -c copy -f segment -segment_time $SEGMENT_TIME -reset_timestamps 1 "/output/chunk-%03d.mp4"

# docker build -f ffmpeg.Dockerfile -t spliter .
# docker run -v <targetoutput>:/output -v <source>:/media spliter
# docker run -v $(pwd)/../output-chunks:/output -v $(pwd)/../media:/media spliter