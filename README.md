## Running locally

This project uses [wasm-pack](https://rustwasm.github.io/docs/wasm-pack/) to compile to webassembly 
and [just](https://just.systems/) to run actions.

Use `just build` in the base directory to build, and `just server` to deploy. 
The site will be available at `localhost:8000`.

## TODO

- better editor
- pre-generated levels list
- better win screen/general UI
  - layered canvases (effects -> UI -> grid?)
  - translucent "glow" around filled-in pipes?
- validity checking for board imports
