from __future__ import annotations

import os
from typing import Optional, Generator
import re
import subprocess
import json

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from pytube import YouTube, Playlist, StreamQuery, Stream
from tqdm import tqdm as ProgressBar
import h5py as h5
import scipy.io.wavfile as wav

from constants import *
from utils import copy

def getVideosFromPlaylist(link: str) -> list[str]:
    c = Playlist(link)
    return list(c.video_urls)

# A class that holds a YTvideo object
class YTVideo:
    def __init__(self, url: str) -> None:
        yt: YouTube = YouTube(url)
        self.url = url
        self.title = yt.title
        self.author = yt.author
        self.desc = yt.description
        self.kw = yt.keywords
        
def getYTVideo(link: str) -> Optional[YTVideo]:
    try:
        return YTVideo(link)
    except Exception:
        return

def getAudio(link: str) -> Optional[NDArray[np.int16]]:
    """Given a link, tries to download the audio file and return an array of number representing the wave intensities of the audio file
    Returns None if the download failed
    """
    temp_mp4 = "temp.mp4"
    temp_wav = "temp.wav"
    
    
    # Get the video
    try:
        yt: YouTube = YouTube(link)
    except:
        print(f"Download failed for {link}: cannot load video")
        return

    stream: Optional[Stream] = yt.streams.filter(only_audio=True, subtype="mp4").order_by("abr").last()
    
    if not stream:
        print(f"Download failed for {link}: {yt.title}")
        return
    
    bitrate = int(stream.abr[:-4])
    
    # Download the audio as an mp4
    stream.download(filename = temp_mp4)
    
    # Convert mp4 to wav
    command = ["ffmpeg", "-i", temp_mp4, "-ab", f"{bitrate}k", "-ac", "2", "-ar", "44100", "-vn", temp_wav]
    subprocess.call(command, stdout = subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.remove(temp_mp4)
    
    # Convert wav to numbers
    fs, data = wavfile.read(temp_wav)
    os.remove(temp_wav)
    
    return data

# Channel sources
class SourceReader:
    def __init__(self, labelDirectory: str):
        with open(labelDirectory) as f:
            strs = f.readlines()
        self.labels = copy(strs)
    
    def __iter__(self):
        for l in self.labels:
            if l.isspace():
                continue
            yield l

# Make preliminary labels using full text match in labels
class LabelMaker(SourceReader):
    def makeLabels(self, title: str, description: str) -> list[str]:
        return []
    TODO

# Splice audios. Return an iterator of numpy arrays and starting times
def spliceAudio(src: NDArray[np.int16]) -> Generator[tuple[NDArray[np.int16], int], None, None]:
    start = 0
    duration = AUDIO_LENGTH * BITRATE
    while start < src.shape[0] - duration:
        trimmed = src[start:start+duration]
        start_time = start
        yield (trimmed, start_time)
        start += SAMPLE_EVERY_SECOND * BITRATE
        

# If overwrite set to true, then update all datasets even if it exists
def makeData(playlists: SourceReader, exclusions: SourceReader, root: str = ".."):
    """Creates the dataset with primitive labelling and save them in root/data/

    Args:
        playlists (SourceReader): list of urls to youtube playlist to train
        exclusions (SourceReader): Exclusions are the audios that we dont want from the videos in playlists either because they are trailers, interviews, or the audio quality is just bad
        root (str): root directory of training data
    """
    # Get list of channels
    # Get all videos
    videoUrls: list[str] = []
    for playlist in playlists:
        for url in getVideosFromPlaylist(playlist):
            if url in exclusions:
                continue
            videoUrls.append(url)
    
    # Index the data. Keep track of the metadatas. We will add labels later in a separate manual process
    dataIndex = 0
    metadatas = {}
    metadatasPath = f"{root}/data"
    if not os.path.exists(metadatasPath): 
        os.makedirs(metadatasPath)
    
    # Loop through every url and download video
    for url in ProgressBar(videoUrls):
        # Get the video metadata
        video = getYTVideo(url)
        if not video:
            continue
        
        # Create the folder for data
        titleAlphanumeric = re.sub(r'\W+', '', video.title)
        dataPath = f"{root}/data/{titleAlphanumeric}"
        if not os.path.exists(dataPath): 
            os.makedirs(dataPath)
        
        # Get the audio data
        audioData = getAudio(url)
        if audioData is None:
            continue
     
        # Normalize audioData later
        normalizer = float(np.max(np.abs(audioData)))
        
        # Write the wav file
        audioPath = f"{dataPath}/audio.wav"
        wav.write(audioPath, BITRATE, audioData)
        
        # Iterate through trimmed audio datas
        for i, (trim, starttime) in enumerate(spliceAudio(audioData)):
            # Ignore silent portions
            if np.abs(trim).mean() < VOLUME_THRESHOLD * normalizer:
                continue
            
            metadata = {
                "title": video.title,
                "sequence": i,
                "path": audioPath,
                "author": video.author,
                "start_time_seconds": starttime,
                "description": video.desc,
                "keywords": video.kw,
                "url": video.url,
                "global_max_volume": normalizer,
                "labels": []
            }
            
            metadatas[dataIndex] = metadata
            
            with open(f"{metadatasPath}/metadata.json", 'w') as f:
                f.write(json.dumps(metadatas, indent = 4))

## Debug
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    audioData = getAudio(url)
    if audioData:
        data = np.array(audioData)
        print(data.shape, np.max(data[:, 0]))