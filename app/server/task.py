from typing import Optional

from pydantic import BaseModel


class DecensorItem(BaseModel):
    img_id: str
    """Images were uploaded earlier."""
    variations: int
    """How many variations to generate"""
    is_moasic: bool
    """Which model to use"""
    mask: str
    """RGB mask or file"""
    output: Optional[str]
    """None will return to web"""