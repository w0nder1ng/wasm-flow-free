build_cmd := "wasm-pack build --no-pack --target web --out-dir www/pkg"
build:
    {{ build_cmd }}

server:
    python3 -m http.server -d www/

sprite_gen:
    #!/bin/sh
    # this command depends on `pillow`, `numpy`, and `pycairo`
    source .venv/bin/activate
    python3 utils/generate_sprites.py -o /tmp/out.png -s 40
    python3 utils/sprite_sheet_to_arr.py -s 40 -n 16 /tmp/out.png