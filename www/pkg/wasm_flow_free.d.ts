/* tslint:disable */
/* eslint-disable */
/**
* @param {Uint8Array} data
*/
export function seed_rng(data: Uint8Array): void;
/**
*/
export class Canvas {
  free(): void;
/**
* @param {number} width
* @param {number} height
* @returns {Canvas}
*/
  static new(width: number, height: number): Canvas;
/**
*/
  render(): void;
/**
* @returns {number}
*/
  get_pix_buf(): number;
/**
* @returns {number}
*/
  width(): number;
/**
* @returns {number}
*/
  height(): number;
/**
* @returns {number}
*/
  canvas_height(): number;
/**
* @returns {number}
*/
  canvas_width(): number;
/**
* @param {Int32Array} pos
*/
  handle_md(pos: Int32Array): void;
/**
*/
  handle_mu(): void;
/**
* @param {Int32Array} pos
*/
  handle_mm(pos: Int32Array): void;
/**
* @param {string} keypress
*/
  handle_keypress(keypress: string): void;
/**
* @returns {boolean}
*/
  check_all_connected(): boolean;
/**
* @returns {boolean}
*/
  game_won(): boolean;
/**
* @returns {Uint8Array}
*/
  write_board(): Uint8Array;
/**
* @param {Uint8Array} serialized
*/
  read_board(serialized: Uint8Array): void;
/**
* @param {number} width
* @param {number} height
* @returns {Canvas}
*/
  static gen_filled_board(width: number, height: number): Canvas;
/**
* @param {number} width
* @param {number} height
* @returns {Canvas}
*/
  static gen_new_board(width: number, height: number): Canvas;
/**
* @returns {Uint8Array}
*/
  to_bytes(): Uint8Array;
/**
* @param {Uint8Array} board
*/
  from_bytes(board: Uint8Array): void;
/**
*/
  hint(): void;
/**
* @param {number} width
* @param {number} height
*/
  resize(width: number, height: number): void;
/**
* @param {number} x
* @param {number} y
* @param {number} color
*/
  add_dot_at(x: number, y: number, color: number): void;
/**
* @param {number} x
* @param {number} y
*/
  remove_dot_at(x: number, y: number): void;
/**
* @param {Uint16Array | undefined} [new_palette]
*/
  remap_color_palette(new_palette?: Uint16Array): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_canvas_free: (a: number) => void;
  readonly canvas_new: (a: number, b: number) => number;
  readonly canvas_render: (a: number) => void;
  readonly canvas_get_pix_buf: (a: number) => number;
  readonly canvas_width: (a: number) => number;
  readonly canvas_height: (a: number) => number;
  readonly canvas_canvas_height: (a: number) => number;
  readonly canvas_canvas_width: (a: number) => number;
  readonly canvas_handle_md: (a: number, b: number, c: number) => void;
  readonly canvas_handle_mu: (a: number) => void;
  readonly canvas_handle_mm: (a: number, b: number, c: number) => void;
  readonly canvas_handle_keypress: (a: number, b: number, c: number) => void;
  readonly canvas_check_all_connected: (a: number) => number;
  readonly canvas_game_won: (a: number) => number;
  readonly canvas_read_board: (a: number, b: number, c: number) => void;
  readonly canvas_gen_filled_board: (a: number, b: number) => number;
  readonly canvas_gen_new_board: (a: number, b: number) => number;
  readonly canvas_to_bytes: (a: number, b: number) => void;
  readonly canvas_from_bytes: (a: number, b: number, c: number) => void;
  readonly canvas_hint: (a: number) => void;
  readonly canvas_resize: (a: number, b: number, c: number) => void;
  readonly canvas_add_dot_at: (a: number, b: number, c: number, d: number) => void;
  readonly canvas_remove_dot_at: (a: number, b: number, c: number) => void;
  readonly canvas_remap_color_palette: (a: number, b: number, c: number) => void;
  readonly seed_rng: (a: number, b: number) => void;
  readonly canvas_write_board: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
