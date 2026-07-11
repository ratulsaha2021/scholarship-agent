"""OCR module - extracts text from images."""

from pathlib import Path
from typing import Optional

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

class OCRProcessor:
    """Extracts text from images using OCR."""
    
    def __init__(self):
        self.available = HAS_OCR
    
    def extract_text(self, image_path: Path) -> str:
        """Extract text from an image file."""
        if not self.available:
            raise ImportError("pytesseract and Pillow required for OCR. Install with: pip install pytesseract Pillow")
        
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}")
    
    def extract_text_from_bytes(self, image_bytes: bytes, filename: str = "image.png") -> str:
        """Extract text from image bytes."""
        if not self.available:
            raise ImportError("pytesseract and Pillow required for OCR")
        
        import io
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.available
