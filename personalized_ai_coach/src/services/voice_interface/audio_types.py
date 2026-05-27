class AudioChunk:
    """Simple container for audio data from WebRTC."""
    def __init__(self, data: bytes, sample_rate: int, encoding: str):
        self.data = data
        self.sample_rate = sample_rate
        self.encoding = encoding
