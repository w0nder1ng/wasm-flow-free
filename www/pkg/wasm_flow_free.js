let wasm;

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8Memory0 = null;

function getUint8Memory0() {
    if (cachedUint8Memory0 === null || cachedUint8Memory0.byteLength === 0) {
        cachedUint8Memory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8Memory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len));
}

function _assertNum(n) {
    if (typeof(n) !== 'number') throw new Error(`expected a number argument, found ${typeof(n)}`);
}

let cachedUint32Memory0 = null;

function getUint32Memory0() {
    if (cachedUint32Memory0 === null || cachedUint32Memory0.byteLength === 0) {
        cachedUint32Memory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32Memory0;
}

let WASM_VECTOR_LEN = 0;

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32Memory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available') } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (typeof(arg) !== 'string') throw new Error(`expected a string argument, found ${typeof(arg)}`);

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8Memory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8Memory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8Memory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);
        if (ret.read !== arg.length) throw new Error('failed to pass whole string');
        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedInt32Memory0 = null;

function getInt32Memory0() {
    if (cachedInt32Memory0 === null || cachedInt32Memory0.byteLength === 0) {
        cachedInt32Memory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32Memory0;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8Memory0().subarray(ptr / 1, ptr / 1 + len);
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8Memory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedUint16Memory0 = null;

function getUint16Memory0() {
    if (cachedUint16Memory0 === null || cachedUint16Memory0.byteLength === 0) {
        cachedUint16Memory0 = new Uint16Array(wasm.memory.buffer);
    }
    return cachedUint16Memory0;
}

function passArray16ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 2, 2) >>> 0;
    getUint16Memory0().set(arg, ptr / 2);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}
/**
* @param {Uint8Array} data
*/
export function seed_rng(data) {
    const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.seed_rng(ptr0, len0);
}

const CanvasFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_canvas_free(ptr >>> 0));
/**
*/
export class Canvas {

    constructor() {
        throw new Error('cannot invoke `new` directly');
    }

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Canvas.prototype);
        obj.__wbg_ptr = ptr;
        CanvasFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CanvasFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_canvas_free(ptr);
    }
    /**
    * @param {number} width
    * @param {number} height
    * @returns {Canvas}
    */
    static new(width, height) {
        _assertNum(width);
        _assertNum(height);
        const ret = wasm.canvas_new(width, height);
        return Canvas.__wrap(ret);
    }
    /**
    */
    render() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.canvas_render(this.__wbg_ptr);
    }
    /**
    * @returns {number}
    */
    get_pix_buf() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_get_pix_buf(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
    * @returns {number}
    */
    width() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
    * @returns {number}
    */
    height() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
    * @returns {number}
    */
    canvas_height() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_canvas_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
    * @returns {number}
    */
    canvas_width() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_canvas_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
    * @param {Int32Array} pos
    */
    handle_md(pos) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passArray32ToWasm0(pos, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.canvas_handle_md(this.__wbg_ptr, ptr0, len0);
    }
    /**
    */
    handle_mu() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        wasm.canvas_handle_mu(this.__wbg_ptr);
    }
    /**
    * @param {Int32Array} pos
    */
    handle_mm(pos) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passArray32ToWasm0(pos, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.canvas_handle_mm(this.__wbg_ptr, ptr0, len0);
    }
    /**
    * @param {string} keypress
    */
    handle_keypress(keypress) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passStringToWasm0(keypress, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.canvas_handle_keypress(this.__wbg_ptr, ptr0, len0);
    }
    /**
    * @returns {boolean}
    */
    check_all_connected() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_check_all_connected(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
    * @returns {boolean}
    */
    game_won() {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ret = wasm.canvas_game_won(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
    * @returns {Uint8Array}
    */
    write_board() {
        try {
            if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertNum(this.__wbg_ptr);
            wasm.canvas_write_board(retptr, this.__wbg_ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_free(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {Uint8Array} serialized
    */
    read_board(serialized) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passArray8ToWasm0(serialized, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.canvas_read_board(this.__wbg_ptr, ptr0, len0);
    }
    /**
    * @param {number} width
    * @param {number} height
    * @returns {Canvas}
    */
    static gen_filled_board(width, height) {
        _assertNum(width);
        _assertNum(height);
        const ret = wasm.canvas_gen_filled_board(width, height);
        return Canvas.__wrap(ret);
    }
    /**
    * @param {number} width
    * @param {number} height
    * @returns {Canvas}
    */
    static gen_new_board(width, height) {
        _assertNum(width);
        _assertNum(height);
        const ret = wasm.canvas_gen_new_board(width, height);
        return Canvas.__wrap(ret);
    }
    /**
    * @returns {Uint8Array}
    */
    to_bytes() {
        try {
            if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertNum(this.__wbg_ptr);
            wasm.canvas_to_bytes(retptr, this.__wbg_ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            var v1 = getArrayU8FromWasm0(r0, r1).slice();
            wasm.__wbindgen_free(r0, r1 * 1, 1);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {Uint8Array} board
    */
    from_bytes(board) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        const ptr0 = passArray8ToWasm0(board, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.canvas_from_bytes(this.__wbg_ptr, ptr0, len0);
    }
    /**
    * @param {number} width
    * @param {number} height
    */
    resize(width, height) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(width);
        _assertNum(height);
        wasm.canvas_resize(this.__wbg_ptr, width, height);
    }
    /**
    * @param {number} x
    * @param {number} y
    * @param {number} color
    */
    add_dot_at(x, y, color) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(x);
        _assertNum(y);
        _assertNum(color);
        wasm.canvas_add_dot_at(this.__wbg_ptr, x, y, color);
    }
    /**
    * @param {number} x
    * @param {number} y
    */
    remove_dot_at(x, y) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        _assertNum(x);
        _assertNum(y);
        wasm.canvas_remove_dot_at(this.__wbg_ptr, x, y);
    }
    /**
    * @param {Uint16Array | undefined} [new_palette]
    */
    remap_color_palette(new_palette) {
        if (this.__wbg_ptr == 0) throw new Error('Attempt to use a moved value');
        _assertNum(this.__wbg_ptr);
        var ptr0 = isLikeNone(new_palette) ? 0 : passArray16ToWasm0(new_palette, wasm.__wbindgen_malloc);
        var len0 = WASM_VECTOR_LEN;
        wasm.canvas_remap_color_palette(this.__wbg_ptr, ptr0, len0);
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, maybe_memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedInt32Memory0 = null;
    cachedUint16Memory0 = null;
    cachedUint32Memory0 = null;
    cachedUint8Memory0 = null;


    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(input) {
    if (wasm !== undefined) return wasm;

    if (typeof input === 'undefined') {
        input = new URL('wasm_flow_free_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof input === 'string' || (typeof Request === 'function' && input instanceof Request) || (typeof URL === 'function' && input instanceof URL)) {
        input = fetch(input);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await input, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync }
export default __wbg_init;
