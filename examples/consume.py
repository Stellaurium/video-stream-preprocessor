"""Run from project root: python -m examples.consume"""
from screenprep.sdk import FrameClient

for packet in FrameClient().frames(fps=15):
    # packet.image is the corrected/cropped uint8 BGR frame, with no JPEG compression.
    print(packet.sequence,packet.image.shape,packet.source_seconds,flush=True)
    # Pass packet.image to OCR / detection / the next pipeline stage here.
