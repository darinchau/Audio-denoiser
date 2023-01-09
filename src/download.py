import os
from typing import Optional
from pytube import YouTube, Playlist, StreamQuery, Stream
import numpy as np
from utils import copy
import subprocess
from scipy.io import wavfile
from typing import Any
from numpy.typing import NDArray

def getVideosFromPlaylist(link: str) -> list[str]:
    c = Playlist(link)
    return list(c.video_urls)

# A class that holds a YTvideo object
class YTVideo:
    def __init__(self, url: str) -> None:
        try:
            yt: YouTube = YouTube(url)
        except Exception:
            return
        
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

## Debug
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    audioData = getAudio(url)
    if audioData:
        data = np.array(audioData)
        print(data.shape, np.max(data[:, 0]))