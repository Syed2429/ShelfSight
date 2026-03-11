"""
Image utility helpers for the gateway service.
Handles encoding/decoding images for inter-service HTTP transport.
"""

import base64
import io
from PIL import Image


def encode_image(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image to a base64 string for JSON transport."""
    # TODO: implement image serialisation
    raise NotImplementedError


def decode_image(b64_string: str) -> Image.Image:
    """Decode a base64 string back to a PIL Image."""
    # TODO: implement image deserialisation
    raise NotImplementedError


def resize_image(image: Image.Image, max_side: int = 1024) -> Image.Image:
    """Proportionally resize an image so its longest side <= max_side."""
    # TODO: implement proportional resize preserving aspect ratio
    raise NotImplementedError
