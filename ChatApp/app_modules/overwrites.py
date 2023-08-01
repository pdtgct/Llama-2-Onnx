from __future__ import annotations
from typing import List, Tuple

from app_modules.presets import gr
from app_modules.utils import detect_converted_mark, convert_asis, convert_mdtext


def postprocess(
    self, y: List[Tuple[str | None, str | None]]
) -> List[Tuple[str | None, str | None]]:
    """
    Parameters:
        y: List of tuples representing the message and response pairs.
        Each message and response should be a string,
        which may be in Markdown format.
    Returns:
        List of tuples representing the message and response.
        Each message and response will be a string of HTML.
    """
    if y is None or y == []:
        return []
    temp = []
    for x in y:
        user, bot = x
        if not detect_converted_mark(user):
            user = convert_asis(user)
        if not detect_converted_mark(bot):
            bot = convert_mdtext(bot)
        temp.append((user, bot))
    return temp


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
