const CACHE_DIR = 'qwen-cache';
const CACHE_VERSION = 'v1';

export const NUM_LAYERS_DIT = 32;
export const NUM_LAYERS_TEXT_ENCODER = 36;

// OPFS name -> source URL.
// const urlFor = (name) => `https://huggingface.co/inventwithdean/Qwen-Image-2.1/resolve/main/${name}`;
const urlFor = (name) => `./out/${name}`;

let dirPromise = null;

async function cacheDir() {
    if (!dirPromise) dirPromise = openCacheDir().catch((e) => { dirPromise = null; throw e; });
    return dirPromise;
}

async function openCacheDir() {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(CACHE_DIR, { create: true });

    let current = null;
    try {
        current = await (await dir.getFileHandle('VERSION')).getFile().then((f) => f.text());
    } catch { /* first run */ }

    if (current !== CACHE_VERSION) {
        console.log(`OPFS cache: version ${current} -> ${CACHE_VERSION}, wiping`);
        const names = [];
        for await (const name of dir.keys()) names.push(name);
        for (const name of names) await dir.removeEntry(name);

        const vh = await dir.getFileHandle('VERSION', { create: true });
        const w = await vh.createWritable();
        await w.write(CACHE_VERSION);
        await w.close();
    }
    return dir;
}

// Returns an OPFS file handle, downloading the file first if needed.
// createWritable() writes to a swap file and commits only on close(), so a
// non-empty file is always a fully downloaded one.
//
// `name` may carry repo subdirectories ("text_encoder/shell.bpk"): it is used
// as the remote path, while the OPFS entry is flattened ('/' -> '__') because
// OPFS file names cannot contain slashes.
export async function ensureCached(name) {
    const dir = await cacheDir();
    const opfsName = name.replaceAll('/', '__');
    const fh = await dir.getFileHandle(opfsName, { create: true });
    if ((await fh.getFile()).size > 0) return fh;

    console.log(`Downloading ${name} -> OPFS`);
    const res = await fetch(urlFor(name));
    if (!res.ok || !res.body) throw new Error(`Failed to fetch ${urlFor(name)}`);
    await res.body.pipeTo(await fh.createWritable()); // closes the writable when done
    return fh;
}

export async function readBytes(name) {
    const file = await (await ensureCached(name)).getFile();
    return new Uint8Array(await file.arrayBuffer());
}

export async function readFloats(name) {
    const file = await (await ensureCached(name)).getFile();
    return new Float32Array(await file.arrayBuffer());
}

export async function openSync(name) {
    return (await ensureCached(name)).createSyncAccessHandle();
}

export async function wipe() {
    dirPromise = null;
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(CACHE_DIR, { recursive: true }).catch(() => { });
    console.log('OPFS cache wiped.');
}
