// rpc.js 
// minimal promise-based RPC over postMessage, shared by page and worker.

// Workers live in a separate realm: the only channel to the page is postMessage,
// and only structured-cloneable data can cross. This turns that raw channel into
// two halves so the rest of the code is just `await api.generate({...})` vs.
// plain exported functions.

// ---- page side -------------------------------------------------------------
export function createClient(worker) {
    let nextId = 1;
    const pending = new Map();
    const listeners = new Map();

    worker.addEventListener('message', ({ data }) => {
        if (data && data.id !== undefined) {
            const call = pending.get(data.id);
            if (!call) return;
            pending.delete(data.id);
            if (data.ok) call.resolve(data.value);
            else call.reject(new Error(data.error));
        } else if (data && data.event) {
            listeners.get(data.event)?.forEach((fn) => fn(data.data));
        }
    });

    worker.addEventListener('error', (e) => {
        const err = new Error(e.message || 'Worker crashed');
        for (const call of pending.values()) call.reject(err);
        pending.clear();
    });

    return {
        call(method, arg, transfer = []) {
            const id = nextId++;
            return new Promise((resolve, reject) => {
                pending.set(id, { resolve, reject });
                worker.postMessage({ id, method, arg }, transfer);
            });
        },
        on(event, fn) {
            if (!listeners.has(event)) listeners.set(event, new Set());
            listeners.get(event).add(fn);
        },
    };
}

// ---- worker side -----------------------------------------------------------
export function notify(event, data) {
    self.postMessage({ event, data });
}

// Handlers take one `arg` and return their result.
export function createServer(handlers) {
    self.addEventListener('message', async ({ data }) => {
        if (!data || data.id === undefined) return; // not an RPC call
        try {
            const handler = handlers[data.method];
            if (!handler) throw new Error(`Unknown method: ${data.method}`);
            const result = await handler(data.arg);
            if (result && typeof result === 'object' && Array.isArray(result.transfer)) {
                self.postMessage({ id: data.id, ok: true, value: result.value }, result.transfer);
            } else {
                self.postMessage({ id: data.id, ok: true, value: result });
            }
        } catch (error) {
            self.postMessage({ id: data.id, ok: false, error: String(error?.stack || error) });
        }
    });
}