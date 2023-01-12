from download import makeData, SourceReader

if __name__ == "__main__":
    playlists = SourceReader("./datasrc/playlist.txt")
    exclusions = SourceReader("./datasrc/exclusions.txt")
    root = ".."
    makeData(playlists, exclusions, root)