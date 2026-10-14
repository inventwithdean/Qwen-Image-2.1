use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Bytes, Device},
};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use crate::{
    qwen_image::QwenImageTransformerModelConfig,
    qwen_image_modules::{QwenImageTransformerBlock, QwenImageTransformerBlockConfig},
    text_encoder::text_encoder_modules::{Qwen3VLTextDecoderLayer, Qwen3VLTextDecoderLayerConfig},
};

#[wasm_bindgen]
extern "C" {
    pub type SyncHandle;

    #[wasm_bindgen(method, js_name = getSize)]
    fn get_size(this: &SyncHandle) -> f64;

    #[wasm_bindgen(method, js_name = read)]
    fn read_at(this: &SyncHandle, buf: &mut [u8], opts: &JsValue) -> f64;

    #[wasm_bindgen(method, js_name = close)]
    pub fn close(this: &SyncHandle);
}

fn read_opts(at: usize) -> JsValue {
    let o = js_sys::Object::new();
    js_sys::Reflect::set(&o, &"at".into(), &JsValue::from_f64(at as f64)).unwrap();
    o.into()
}

// Synchronously read the whole file behind a handle into wasm memory
pub fn read_bytes(handle: &SyncHandle) -> Vec<u8> {
    let size = handle.get_size() as usize;
    let mut buf = vec![0u8; size];

    let mut off = 0;
    while off < size {
        let n = handle.read_at(&mut buf[off..], &read_opts(off)) as usize;
        assert!(n > 0, "OPFS read returned 0 bytes (offset {off}/{size})");
        off += n;
    }

    buf
}

/// Reads record without closing the handle
fn read_record(handle: &SyncHandle) -> ModuleRecord {
    ModuleRecord::from_bytes(Bytes::from_bytes_vec(read_bytes(handle)))
        .unwrap_or_else(|e| panic!("bad record: {e:?}"))
}

/// Copies bytes from the handle to wasm memory and closes the handle
pub fn read_and_close(handle: SyncHandle) -> Vec<u8> {
    let bytes = read_bytes(&handle);
    handle.close();
    bytes
}

/// Extracts record and closes the handle
pub fn take_record(handle: SyncHandle) -> Result<ModuleRecord, String> {
    let bytes = read_and_close(handle);
    ModuleRecord::from_bytes(Bytes::from_bytes_vec(bytes)).map_err(|e| e.to_string())
}

pub struct QwenImageBlockStreamerWeb {
    pub num_layers: usize,
    // Handles for transformer blocks
    handles: Vec<SyncHandle>,
    active_block: Option<QwenImageTransformerBlock>,
}

impl QwenImageBlockStreamerWeb {
    pub fn new_empty(config: &QwenImageTransformerModelConfig, device: &Device) -> Self {
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let block_config = QwenImageTransformerBlockConfig::new(
            inner_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            config.mlp_ratio,
            config.eps,
        );

        Self {
            num_layers: config.num_layers,
            handles: Vec::with_capacity(config.num_layers),
            active_block: Some(block_config.init(device)),
        }
    }

    // Register the OPFS file for the next layer (must be called in layer order).
    pub fn add_handle(&mut self, handle: SyncHandle) -> Result<usize, String> {
        if self.handles.len() == self.num_layers {
            return Err("All handles already registered".to_string());
        }
        self.handles.push(handle);
        Ok(self.handles.len() - 1)
    }

    pub fn is_ready(&self) -> bool {
        self.handles.len() == self.num_layers
    }

    pub fn block_size(&self, layer: usize) -> usize {
        self.handles[layer].get_size() as usize
    }

    pub fn get(&mut self, layer: usize) -> &QwenImageTransformerBlock {
        assert!(
            self.is_ready(),
            "only {}/{} block handles registered",
            self.handles.len(),
            self.num_layers
        );

        let record = read_record(&self.handles[layer]);

        let block = self
            .active_block
            .take()
            .expect("Active block not initialized");
        self.active_block = Some(block.load_record(record));
        self.active_block.as_ref().unwrap()
    }

    pub fn close_all(&mut self) {
        for h in self.handles.drain(..) {
            h.close();
        }
    }
}

pub struct QwenImageTextEncoderStreamerWeb {
    pub num_layers: usize,
    // Handles for decoder layers
    handles: Vec<SyncHandle>,
    active_block: Option<Qwen3VLTextDecoderLayer>,
}

impl QwenImageTextEncoderStreamerWeb {
    pub fn new_empty(device: &Device) -> Self {
        let num_layers = 36;
        let block = Qwen3VLTextDecoderLayerConfig::new(4096, 128, 32, 8, 12288, 1e-6).init(device);

        Self {
            num_layers,
            handles: Vec::with_capacity(num_layers),
            active_block: Some(block),
        }
    }

    pub fn add_handle(&mut self, handle: SyncHandle) -> Result<usize, String> {
        if self.handles.len() == self.num_layers {
            return Err("All handles already registered!".to_string());
        }
        self.handles.push(handle);
        Ok(self.handles.len() - 1)
    }

    pub fn is_ready(&self) -> bool {
        self.handles.len() == self.num_layers
    }

    pub fn block_size(&self, layer: usize) -> usize {
        self.handles[layer].get_size() as usize
    }

    pub fn get(&mut self, layer: usize) -> &Qwen3VLTextDecoderLayer {
        assert!(
            self.is_ready(),
            "only {}/{} block handles registered",
            self.handles.len(),
            self.num_layers
        );

        let record = read_record(&self.handles[layer]);

        let block = self
            .active_block
            .take()
            .expect("Active block not initialized");
        self.active_block = Some(block.load_record(record));
        self.active_block.as_ref().unwrap()
    }

    pub fn close_all(&mut self) {
        for h in self.handles.drain(..) {
            h.close();
        }
    }
}
