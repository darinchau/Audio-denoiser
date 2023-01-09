import os
import numpy as np
from download import getAudio, getVideosFromChannel, getYTVideo
import json
import h5py as h5
from numpy.typing import NDArray
from typing import Generator

# Hyperparams for training and testing
# Bitrate of audio
BITRATE = 44100
# If the average of the snippet is too small probably it is a silent portion, skip it
VOLUME_THRESHOLD = 0.2
# Take sample every n seconds
SAMPLE_EVERY_SECOND = 2
# The length of audio to splice the sources in
AUDIO_LENGTH = 15

# Channel sources
def getChannels() -> list[str]:
    with open("./datasrc/channels.txt") as f:
        strs = f.readlines()
    return [s[:-1] if s[-1] == "\n" else s for s in strs]

# Exclusions are the audios that we dont want from the channels in getChannels()
# either because they are trailers, interviews, or the audio quality is just bad
def getExclusions() -> list[str]:
    with open("./datasrc/exclusions.txt") as f:
        strs = f.readlines()
    return [s[:-1] if s[-1] == "\n" else s for s in strs]

# Splice audios. Return an iterator of numpy arrays
def spliceAudio(src: NDArray[np.floating]):
    start = 0
    duration = AUDIO_LENGTH * BITRATE
    while start < src.shape[0] - duration:
        trimmed = src[start:start+duration]
        start_time = start/BITRATE
        yield (trimmed, start_time)
        start += SAMPLE_EVERY_SECOND * BITRATE

# If overwrite set to true, then update all datasets even if it exists
def makeData(overwrite = False):
    # Get list of channels
    # Get all videos
    exclusions = getExclusions()
    videos: list[str] = []
    for channel in getChannels():
        for url in getVideosFromChannel(channel):
            if url in exclusions:
                continue
            videos.append(url)
    
    for url in videos:
        audioData = getAudio(url)
        video = getYTVideo(url)
        if not audioData or not video:
            continue
        
        # Create the folder for data
        dataPath = f"../data/{video.title}"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
     
        # Normalize audioData
        audioData = audioData / np.max(audioData)
        
        for i, (trim, starttime) in enumerate(spliceAudio(audioData)):
            if trim.mean() < VOLUME_THRESHOLD:
                continue
            
            metadata = {
                "title": video.title,
                "author": video.author,
                "start_time_seconds": starttime,
                "description": video.desc,
                "keywords": video.kw,
                "url": video.url,
                "labels": []
            }
            
            json_dir = f"{dataPath}/metadata{i}.json"
            h5dir = f"{dataPath}/audiodata{i}.h5"
            
            with open(json_dir, 'w') as f:
                f.write(json.dumps(metadata, indent = 4))
            
            with h5.File(h5dir, "w") as f:
                dset = f.create_dataset("audio", data = trim)
        
if __name__ == "__main__":
    print(getChannels())
    print(getExclusions())