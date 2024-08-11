## Running locally

This project uses [wasm-pack](https://rustwasm.github.io/docs/wasm-pack/) to compile to webassembly.

Use `wasm-pack build --target web --out-dir www/pkg`
in the base directory to build,
and `python3 -m http.server` in `www/` to deploy.
The site will be available at `localhost:8000`.

## TODO

- move logic to js file or template so it's consistent across pages
- better editor
- pre-generated levels list
- better win screen/general UI
  - layered canvases (effects -> UI -> grid?)
  - translucent "glow" around filled-in pipes?
- looser tracking for mouse position
- validity checking for board imports
- PRNG with seed for puzzle generation?
