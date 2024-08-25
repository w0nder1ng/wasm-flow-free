from base64 import urlsafe_b64decode
from io import BytesIO
from urllib.parse import parse_qs, urlparse

from PIL import Image

query = parse_qs(urlparse(open("board_link.txt").read()).query)
print(query)
b = urlsafe_b64decode(
    query["board"][0],
)

w = int.from_bytes(b[:4])
h = int.from_bytes(b[4:8])
img = Image.frombytes("L", (w*2,h), b[8:])
img.save("tmp.png")
buf = BytesIO()
img.save(buf, format="PNG")
print(f"len {len(buf.getvalue())}")
