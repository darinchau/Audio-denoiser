import os
import numpy as np
from download import getAudio, getVideosFromPlaylist, getYTVideo
import json
import h5py as h5
from numpy.typing import NDArray
from typing import Generator
from tqdm import tqdm as ProgressBar
import re
import scipy.io.wavfile as wav

# Hyperparams for training and testing
# Bitrate of audio
BITRATE = 44100
# If the average of the snippet is too small probably it is a silent portion, skip it
VOLUME_THRESHOLD = 0.01
# Take sample every n seconds
SAMPLE_EVERY_SECOND = 10
# The length of audio to splice the sources in
AUDIO_LENGTH = 10

# Channel sources
def getPlaylists() -> list[str]:
    with open("./datasrc/playlist.txt") as f:
        strs = f.readlines()
    return [s[:-1] if s[-1] == "\n" else s for s in strs]

# Exclusions are the audios that we dont want from the channels in getChannels()
# either because they are trailers, interviews, or the audio quality is just bad
def getExclusions() -> list[str]:
    with open("./datasrc/exclusions.txt") as f:
        strs = f.readlines()
    return [s[:-1] if s[-1] == "\n" else s for s in strs]

# Splice audios. Return an iterator of numpy arrays
def spliceAudio(src: NDArray[np.int16]):
    start = 0
    duration = AUDIO_LENGTH * BITRATE
    while start < src.shape[0] - duration:
        trimmed = src[start:start+duration]
        start_time = start
        yield (trimmed, start_time)
        start += SAMPLE_EVERY_SECOND * BITRATE

# If overwrite set to true, then update all datasets even if it exists
def makeData(playlists, exclusions) -> Generator[tuple[int, dict[str, str | list[str]], NDArray[np.int16], float], None, None]:
    # Get list of channels
    # Get all videos
    videos: list[str] = []
    for playlist in playlists:
        for url in getVideosFromPlaylist(playlist):
            if url in exclusions:
                continue
            videos.append(url)
    
    for url in ProgressBar(videos):
        # Get the video metadata
        video = getYTVideo(url)
        if not video:
            continue
        
        # Create the folder for data
        title_alphanumeric = re.sub(r'\W+', '', video.title)
        dataPath = f"../data/{title_alphanumeric}"
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        
        # Get the audio data
        audioData = getAudio(url)
        if audioData is None:
            continue
     
        # Normalize audioData
        normalizer = float(np.max(np.abs(audioData)))
        
        for i, (trim, starttime) in enumerate(spliceAudio(audioData)):
            # Ignore silent portions
            if np.abs(trim).mean() < VOLUME_THRESHOLD * normalizer:
                continue
            
            metadata = {
                "title": video.title,
                "author": video.author,
                "start_time_seconds": starttime,
                "description": video.desc,
                "keywords": video.kw,
                "url": video.url,
                "max_volume": np.max(np.abs(audioData)),
                "labels": []
            }
            
            newData = np.array(trim)
            
            yield (i, metadata, newData, normalizer)
