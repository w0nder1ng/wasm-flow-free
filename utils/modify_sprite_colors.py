import numpy as np

# from matplotlib import pyplot as plt
from PIL import Image

img = np.array(Image.open("flow_free_sprites2.png"))
print(img)
y_chunks = ((None, 40),)
x_chunks = ((40, 200), (240, None))
for y_chunk in y_chunks:
    for x_chunk in x_chunks:
        print(y_chunk, x_chunk)
        img[slice(*y_chunk), slice(*x_chunk)] = np.clip(
            img[slice(*y_chunk), slice(*x_chunk)], 75, 255,
        )
# plt.imshow(img)
# plt.show()
out = Image.fromarray(img, mode="L")
out.save("flow_free_sprites3.png")
