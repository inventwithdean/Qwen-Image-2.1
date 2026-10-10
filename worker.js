// talks to the page only through rpc.js.
import init, { QwenWeb } from './pkg/qwen_image.js';
import { createServer, notify } from './rpc.js';
import * as store from './store.js';

const TOKEN_PIXELS = 16;

let qwen = null;
let bootPromise = null;
let tokenizerLoaded = false;

const nativeLog = console.log.bind(console);
console.log = (...args) => {
    nativeLog(...args);
    notify('log', args.map((a) => (typeof a === 'string' ? a : String(a))).join(' '));
};

let wasm;
const mem = () => console.log(`wasm heap: ${(wasm.memory.buffer.byteLength / 1048576) | 0} MB`);

async function boot() {
    wasm = await init();
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

async function loadTokenizer() {
    await boot();
    if (!tokenizerLoaded) {
        console.log('Loading tokenizer.json...');
        const bytes = await store.readBytes('tokenizer.json');
        qwen.load_tokenizer(bytes);
        tokenizerLoaded = true;
        console.log('Tokenizer ready.');
    }
}

async function loadTextEncoder() {
    await boot();
    await qwen.load_text_encoder_shell(await store.readBytes('text_encoder/shell.bpk'));

    console.log(`Registering ${store.NUM_LAYERS_TEXT_ENCODER} text encoder layers from OPFS...`);
    for (let i = 0; i < store.NUM_LAYERS_TEXT_ENCODER; i++) {
        const sh = await store.openSync(`text_encoder/block_${i}.bpk`);
        qwen.add_text_encoder_block_handle(sh);
    }
    console.log('Text encoder ready (layers stream from disk on demand).');
}

async function encodePrompt(prompt) {
    await loadTokenizer();
    await loadTextEncoder();
    console.log('Encoding prompt with the text encoder...');
    return await qwen.encode_prompt(prompt); 
}

async function loadShell() {
    await boot();
    await qwen.load_dit_shell(await store.readBytes('shell.bpk'));

    const layers = qwen.num_dit_layers();
    if (layers !== store.NUM_LAYERS_DIT) {
        throw new Error(`Shell exposes ${layers} layers, but store.NUM_LAYERS_DIT is ${store.NUM_LAYERS_DIT}`);
    }

    console.log(`Registering ${store.NUM_LAYERS_DIT} blocks from OPFS...`);
    for (let i = 0; i < store.NUM_LAYERS_DIT; i++) {
        const sh = await store.openSync(`block_${i}.bpk`);
        qwen.add_dit_block_handle(sh);
    }
    console.log('Model ready (blocks stream from disk on demand).');
}


async function generate({ prompt, widthPx, heightPx, steps }) {
    try {
        await boot();

        // Text encoder
        const promptEmbeds = await encodePrompt(prompt);
        qwen.release_text_encoder();
        console.log('Text encoder memory released.');
        // mem();

        // DiT
        const h = Math.round(heightPx / TOKEN_PIXELS); 
        const w = Math.round(widthPx / TOKEN_PIXELS);

        console.log(`Generating ${h}x${w} tokens, ${steps} steps...`);
        await loadShell();
        
        const t0 = performance.now();
        const latents = await qwen.generate(promptEmbeds, h, w, steps);
        console.log(`Denoising done in ${((performance.now() - t0) / 1000).toFixed(2)}s`);
        
        qwen.release_dit();
        // mem();

        // VAE decode
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
            try { qwen.release_text_encoder(); } catch { }
            try { qwen.release_dit(); } catch { }
            try { qwen.release_vae(); } catch { }
        }
    }
}

async function clearCache() {
    if (qwen) {
        try { qwen.release_text_encoder(); } catch { }
        try { qwen.release_dit(); } catch { }
        try { qwen.release_vae(); } catch { }
    }
    await store.wipe();
}

createServer({ generate, clearCache });