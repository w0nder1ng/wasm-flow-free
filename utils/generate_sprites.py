import argparse
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from math import pi

import cairo
from PIL import Image

GRID_H = 1
GRID_W = 16

SPRITE_SIZE = 40
CIRCLE_LARGE = 0.75
CONNECTIONS_WIDTH = 0.5
CIRCLE_SMALL = CONNECTIONS_WIDTH
BG_OPACITY = 0.4
bg_fill = (BG_OPACITY, BG_OPACITY, BG_OPACITY)


@contextmanager
def saved(cr):
    cr.save()
    try:
        yield cr
    finally:
        cr.restore()


@dataclass
class Sprite:
    background_fill: tuple[float, float, float] | None
    circle_size: float | None
    # right, up, left, down (angle order)
    connections: Sequence[bool] | None
    connections_width: float | None

    def render(self, ctx: cairo.Context):
        # probably is already a checkerboard
        ctx.set_source_rgb(0, 0, 0)
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()
        if self.background_fill is not None:
            ctx.set_source_rgb(*self.background_fill)
            ctx.rectangle(0, 0, 1, 1)
            ctx.fill()
        if self.circle_size is not None:
            ctx.set_source_rgb(1, 1, 1)
            ctx.arc(0.5, 0.5, self.circle_size / 2, 0, pi * 2)
            ctx.fill()
        if self.connections is not None:
            ctx.set_source_rgb(1, 1, 1)
            connections_width = self.connections_width or 0.25
            for i, conn in enumerate(self.connections):
                if conn:
                    with saved(ctx):
                        # translate such that 0.5, 0.5 is the new center
                        ctx.translate(0.5, 0.5)
                        angle = pi / 2.0 * i
                        # because the axes are being rotated and not the drawn objects,
                        # we need to negate the angle to get an equivalent rotation
                        ctx.rotate(-angle)
                        # colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
                        # ctx.set_source_rgb(*colors[i])
                        ctx.set_source_rgb(1, 1, 1)
                        start = -connections_width / 2
                        ctx.rectangle(0, start, 0.5, connections_width)
                        ctx.fill()


def main():
    parser = argparse.ArgumentParser(description="generate sprites for flow free")
    parser.add_argument("-o", "--output", type=argparse.FileType("wb"), required=True)
    parser.add_argument("-s", "--sprite-size", type=int, default=SPRITE_SIZE)
    parser.add_argument("-B", "--background-opacity", type=float, default=BG_OPACITY)
    args = parser.parse_args()
    # print(args.output)
    dims = (args.sprite_size * GRID_W, args.sprite_size * GRID_H)
    bg_fill = (args.background_opacity,) * 3
    sprites = [
        Sprite(None, CIRCLE_LARGE, None, None),
        Sprite(bg_fill, CIRCLE_LARGE, [False, False, True, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_LARGE, [True, False, False, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_LARGE, [False, True, False, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_LARGE, [False, False, False, True], CONNECTIONS_WIDTH),
        Sprite(None, None, None, CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [False, False, True, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [True, False, False, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [False, True, False, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [False, False, False, True], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [True, True, False, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [True, False, False, True], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [False, False, True, True], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [False, True, True, False], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [False, True, False, True], CONNECTIONS_WIDTH),
        Sprite(bg_fill, CIRCLE_SMALL, [True, False, True, False], CONNECTIONS_WIDTH),
    ]
    with cairo.ImageSurface(cairo.FORMAT_ARGB32, *dims) as surface:
        context = cairo.Context(surface)
        # context.get_source().set_filter(cairo.FILTER_NEAREST)
        context.scale(*dims)
        # fill with checkerboard
        for r in range(GRID_H * 2):
            for c in range(GRID_W * 2):
                if (r + c) % 2 == 0:
                    context.set_source_rgb(0, 0, 0)
                else:
                    context.set_source_rgb(0.5, 0.5, 0.5)
                with saved(context):
                    context.translate(float(c) / (GRID_W * 2), float(r) / (GRID_H * 2))
                    context.scale(1.0 / (GRID_W * 2), 1.0 / (GRID_H * 2))
                    context.rectangle(0, 0, 1, 1)
                    context.fill()
        # render sprites
        assert len(sprites) <= GRID_W * GRID_H
        for i, sprite in enumerate(sprites):
            c = i % GRID_W
            r = i // GRID_W
            with saved(context):
                context.translate(float(c) / GRID_W, float(r) / GRID_H)
                context.scale(1.0 / GRID_W, 1.0 / GRID_H)
                sprite.render(context)
        out = BytesIO()
        surface.write_to_png(out)
        img = Image.open(out).convert("L")
        img.save(args.output)


if __name__ == "__main__":
    main()
