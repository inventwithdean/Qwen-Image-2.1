import { createClient } from './rpc.js';

const $ = (id) => document.getElementById(id);
const canvas = $('preview');
const ctx = canvas.getContext('2d');
const statusEl = $('status');
const logsEl = $('logs');
const runBtn = $('run');
const saveBtn = $('save');

const stepsSlider = $('steps');
const stepsOut = $('steps-out');
const sizeSelector = $('size-selector');

const worker = new Worker('worker.js', { type: 'module' });
const api = createClient(worker);

api.on('log', (line) => {
    logsEl.textContent += line + '\n';
    logsEl.scrollTop = logsEl.scrollHeight;
    console.log('[worker]', line);
});

stepsSlider.addEventListener('input', refreshSteps);
refreshSteps();

function refreshSteps() {
    const pct = ((stepsSlider.value - stepsSlider.min) / (stepsSlider.max - stepsSlider.min)) * 100;
    stepsSlider.style.setProperty('--fill', `${pct}%`);
    stepsOut.value = stepsSlider.value;
}

function setBusy(busy) {
    runBtn.disabled = busy;
    document.body.classList.toggle('busy', busy);
    stepsSlider.disabled = busy;
    sizeSelector.disabled = busy;
}

function showImage({ width, height, rgba }) {
    canvas.width = width; 
    canvas.height = height;
    
    canvas.style.maxWidth = '100%';
    canvas.style.maxHeight = '100%'; 
    canvas.style.objectFit = 'contain';

    ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
}

runBtn.addEventListener('click', async () => {
    setBusy(true);
    saveBtn.disabled = true;
    statusEl.textContent = 'Working...';
    
    const [widthPx, heightPx] = sizeSelector.value.split('x').map(Number);
    
    try {
        const image = await api.call('generate', {
            widthPx,
            heightPx,
            steps: Number(stepsSlider.value),
        });
        showImage(image);
        statusEl.textContent = `Done — ${image.width}x${image.height}`;
        saveBtn.disabled = false;
    } catch (e) {
        statusEl.textContent = 'Failed';
        console.error(e);
    } finally {
        setBusy(false);
    }
});

saveBtn.addEventListener('click', () => {
    canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qwen_output_${sizeSelector.value}.png`;
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10_000);
    }, 'image/png');
});