import init, { QwenWeb } from './pkg/qwen_image.js';

const NUM_LAYERS = 32;
// Bump this whenever re-exporting the .bpk files; a mismatch wipes the OPFS cache.
const CACHE_VERSION = 'v1';

let qwenModel = null; // survives across runs; sync handles are exclusive locks,
let syncHandles = []; // so we open them once and keep them for the worker's life
let loadPromise = null;

async function fetchBinary(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}`);
    return new Uint8Array(await res.arrayBuffer());
}

async function fetchFloatArray(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}`);
    return new Float32Array(await res.arrayBuffer());
}

async function getCacheDir() {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle('qwen-cache', { create: true });

    // Version check
    let current = null;
    try {
        const f = await (await dir.getFileHandle('VERSION')).getFile();
        current = await f.text();
    } catch { /* first run */ }

    if (current !== CACHE_VERSION) {
        console.log(`Cache version ${current} -> ${CACHE_VERSION}, clearing OPFS cache`);
        for await (const name of dir.keys()) {
            await dir.removeEntry(name);
        }
        const vh = await dir.getFileHandle('VERSION', { create: true });
        const w = await vh.createWritable();
        await w.write(CACHE_VERSION);
        await w.close();
    }
    return dir;
}

// Download straight into OPFS (streamed, never fully in RAM).
// createWritable writes to a swap file and only commits on close(),
// so a non-empty file is always a complete one.
async function ensureCached(dir, name, url) {
    const fh = await dir.getFileHandle(name, { create: true });
    if ((await fh.getFile()).size > 0) return fh;

    console.log(`Downloading ${name} -> OPFS`);
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`Failed to fetch ${url}`);
    const writable = await fh.createWritable();
    await res.body.pipeTo(writable); // closes the writable when done
    return fh;
}

async function loadModel() {
    await init();
    console.log("WASM loaded. Initializing Burn...");

    const model = new QwenWeb();
    const dir = await getCacheDir();

    // Shell is small enough to pass through RAM once
    const shellBytes = await fetchBinary('./out/shell.bpk');
    await model.load_shell(shellBytes);

    console.log(`Caching ${NUM_LAYERS} blocks in OPFS...`);
    const handles = [];
    try {
        for (let i = 0; i < NUM_LAYERS; i++) {
            const fh = await ensureCached(dir, `block_${i}.bpk`, `./out/block_${i}.bpk`);
            const sh = await fh.createSyncAccessHandle(); // exclusive lock, worker-only
            handles.push(sh);
            model.add_block_handle(sh); // layer order == registration order
        }
    } catch (e) {
        for (const h of handles) { try { h.close(); } catch { } }
        throw e;
    }

    qwenModel = model;
    syncHandles = handles;
    console.log("All blocks registered (streamed from disk on demand).");
}

async function clearCache() {
    if (qwenModel) {
        qwenModel.close_blocks();
        qwenModel = null;
        syncHandles = [];
        loadPromise = null;
    }
    const root = await navigator.storage.getDirectory();
    await root.removeEntry('qwen-cache', { recursive: true }).catch(() => { });
    console.log("OPFS cache cleared. Reload the page before running again.");
}

self.onmessage = async (event) => {
    const { type } = event.data;

    if (type === 'CLEAR_CACHE') {
        try {
            await clearCache();
            self.postMessage({ type: 'CACHE_CLEARED' });
        } catch (error) {
            self.postMessage({ type: 'ERROR', message: error.toString() });
        }
        return;
    }

    if (type === 'START_INFERENCE') {
        console.log("Worker received start signal!");
        try {
            if (!loadPromise) {
                loadPromise = loadModel().catch((e) => { loadPromise = null; throw e; });
            }
            await loadPromise;

            console.log("Loading prompt...");
            const promptEmbeds = await fetchFloatArray('./prompt_embeds.bin');

            console.log("Starting WebGPU inference...");
            const startTime = performance.now();
            const latentsOut = await qwenModel.generate(promptEmbeds);
            console.log(`Inference completed in ${((performance.now() - startTime) / 1000).toFixed(2)}s`);

            self.postMessage({ type: 'DONE', latents: latentsOut }, [latentsOut.buffer]);
        } catch (error) {
            console.error("Inference Error:", error);
            self.postMessage({ type: 'ERROR', message: error.toString() });
        }
    }
};