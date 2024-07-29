## Compiling

This project uses [wasm-pack](https://rustwasm.github.io/docs/wasm-pack/) to compile to webassembly

`wasm-pack build --target web --out-dir www/pkg` to build


## TODO

- move logic to js file or template so it's consistent across pages
- better editor
- pre-generated levels list
- better win screen/general UI
- looser tracking for mouse position
- effective serialization for very large boards
- validity checking for board imports
