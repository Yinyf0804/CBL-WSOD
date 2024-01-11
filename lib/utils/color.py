# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np


Color = {
        'red' : (0, 0, 255),
        'green' : (0, 255, 0),
        'blue' : (255, 0, 0),
        'cyan' : (255, 255, 0),
        'yellow' : (0, 255, 255),
        'magenta' : (255, 0, 255),
        'white' : (255, 255, 255),
        'black' : (0, 0, 0),
}


def color_val(color):
    """Convert various input to color tuples.
    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color]
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')