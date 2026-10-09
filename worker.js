// talks to the page only through rpc.js.
import init, { QwenWeb } from './pkg/qwen_image.js';
import { createServer, notify } from './rpc.js';
import * as store from './store.js';

const TOKEN_PIXELS = 16;

let qwen = null;
let bootPromise = null;
let shellLoaded = false; // shell + streamer + all block handles registered
let syncHandles = [];    // JS-side refs to the OPFS sync handles

// Forward console output including the Rust log() from lib.rs
// ("Completed step N") to the page's log panel.
const nativeLog = console.log.bind(console);
console.log = (...args) => {
    nativeLog(...args);
    notify('log', args.map((a) => (typeof a === 'string' ? a : String(a))).join(' '));
};

async function boot() {
    if (!bootPromise) {
        bootPromise = (async () => {
            await init();
            console.log('WASM loaded, initializing WebGPU device...');
            qwen = new QwenWeb();
            await qwen.init_device();
            console.log('Device ready.');
        })().catch((e) => { bootPromise = null; throw e; });
    }
    return bootPromise;
}

// (Re)load the shell and register the block handles. Runs once, then again
// after every run, because release_dit() drops the streamer and its handles.
async function loadShell() {
    await boot();
    await qwen.load_shell(await store.readBytes('shell.bpk'));

    const layers = qwen.num_layers();
    if (layers !== store.NUM_LAYERS) {
        throw new Error(`Shell exposes ${layers} layers, but store.NUM_LAYERS is ${store.NUM_LAYERS}`);
    }

    console.log(`Registering ${store.NUM_LAYERS} blocks from OPFS...`);
    const handles = [];
    try {
        for (let i = 0; i < store.NUM_LAYERS; i++) {
            const sh = await store.openSync(`block_${i}.bpk`); // exclusive lock, worker-only
            handles.push(sh);
            qwen.add_block_handle(sh); // layer order == registration order
        }
    } catch (e) {
        for (const h of handles) { try { h.close(); } catch { } }
        throw e;
    }

    syncHandles = handles;
    shellLoaded = true;
    console.log('Model ready (blocks stream from disk on demand).');
}

async function generate({ widthPx, heightPx, steps }) {
    try {
        await boot();
        if (!shellLoaded) await loadShell();

        const h = Math.round(heightPx / TOKEN_PIXELS); // token grid
        const w = Math.round(widthPx / TOKEN_PIXELS);

        console.log(`Generating ${h}x${w} tokens, ${steps} steps...`);
        const promptEmbeds = await store.readFloats('prompt_embeds.bin');

        const t0 = performance.now();
        const latents = await qwen.generate(promptEmbeds, h, w, steps);
        console.log(`Denoising done in ${((performance.now() - t0) / 1000).toFixed(2)}s`);

        qwen.release_dit(); // frees the DiT, closes the block handles
        shellLoaded = false;

        console.log('Decoding latents with VAE...');
        
        const vaeBytes = await store.readBytes('vae.bpk');
        const image = await qwen.decode_latents(latents, vaeBytes, h, w);

        const rgba = image.rgba;
        const value = { width: image.width, height: image.height, rgba };
        if (typeof image.dispose === 'function') image.dispose();
        else if (typeof image.free === 'function') image.free();
        return { value, transfer: [rgba.buffer] };
    } finally {
        if (qwen) {
            try { qwen.release_dit(); } catch { }
            try { qwen.release_vae(); } catch { }
        }
        for (const h of syncHandles) { try { h.close(); } catch { } }
        syncHandles = [];
        shellLoaded = false;
    }
}

async function clearCache() {
    if (qwen) {
        try { qwen.release_dit(); } catch { }
        try { qwen.release_vae(); } catch { }
    }
    for (const h of syncHandles) { try { h.close(); } catch { } }
    syncHandles = [];
    shellLoaded = false;
    await store.wipe();
}

createServer({ generate, clearCache });