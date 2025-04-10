from pathlib import Path

import PIL.Image


def make_content(role, *contents) -> dict:
    """ :param role: user, assistant, system"""
    # text only
    if isinstance(contents[0], str):
        ret = contents[0]
    # multiple contents
    else:
        ret = []
        for t, v in contents:
            if t == "image":
                if isinstance(v, Path): v = f"file://{v}"
                assert isinstance(v, PIL.Image.Image) or any(v.startswith(prefix) for prefix in ("http", "file://"))
            elif t == "text":
                assert isinstance(v, str)
            else:
                raise TypeError(f"Unsupported content type: {t}")
            ret.append({"type": t, t: v})
    return {"role": role, "content": ret}
