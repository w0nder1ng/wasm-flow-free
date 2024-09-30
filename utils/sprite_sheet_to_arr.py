import argparse

import numpy as np
from PIL import Image


def sprite_sheet_to_arr(file_path, sprite_size, sprite_count=None):
    img = Image.open(file_path)
    img_width, img_height = img.size
    sprite_width, sprite_height = sprite_size
    sprites = []

    for y in range(0, img_height, sprite_height):
        for x in range(0, img_width, sprite_width):
            if sprite_count is not None and len(sprites) >= sprite_count:
                break
            # img.crop((x, y, x + sprite_width, y + sprite_height)).show()
            sprite = (
                np.array(
                    img.crop((x, y, x + sprite_width, y + sprite_height)),
                    dtype=np.uint8,
                )
                .flatten()
                .tolist()
            )
            sprites.append(sprite)
    return sprites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=argparse.FileType("rb"))
    parser.add_argument("-s", "--sprite-size", type=int, required=True)
    parser.add_argument("-n", "--num-sprites", type=int, required=True)
    args = parser.parse_args()
    sprite_size = (args.sprite_size, args.sprite_size)
    sprites = sprite_sheet_to_arr(args.file_path, sprite_size, args.num_sprites)
    for i, sprite in enumerate(sprites):
        if i >= args.num_sprites:
            print("overshot number of sprites")
            break
        with open(f"src/sprites/{i}", "wb") as f:
            f.write(bytearray(sprite))


if __name__ == "__main__":
    main()
