const get = id => document.getElementById(id);
const means = [.485, .456, .406];
const deviations = [.229, .224, .225];
const side = 224;
const plane = side * side;
let token = 0;
let session;

function card(item, match) {
    const figure = document.createElement("figure");
    if (match) figure.classList.add("hit");
    const image = new Image();
    image.src = item.image;
    image.alt = item.label;
    const label = document.createElement("figcaption");
    label.textContent = item.label;
    if (match) {
        const tag = document.createElement("small");
        tag.textContent = "Same product";
        label.append(tag);
    }
    figure.append(image, label);
    return figure;
}

async function search(source, knownId, gallery) {
    const request = ++token;
    get("results").replaceChildren();
    get("status").textContent = "Encoding image…";
    const image = new Image();
    image.src = source;
    try {
        await image.decode();
        if (request !== token) return;
        get("query").src = source;
        get("query").hidden = false;
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = side;
        const context = canvas.getContext("2d");
        context.fillStyle = "#eaf0f7";
        context.fillRect(0, 0, side, side);
        const size = Math.min(side / image.width, side / image.height);
        const width = image.width * size;
        const height = image.height * size;
        context.drawImage(image, (side - width) / 2, (side - height) / 2, width, height);
        const rgba = context.getImageData(0, 0, side, side).data;
        const pixels = new Float32Array(3 * plane);
        for (let c = 0; c < 3; c++) {
            for (let i = 0; i < plane; i++) pixels[c * plane + i] = (rgba[i * 4 + c] / 255 - means[c]) / deviations[c];
        }
        ort.env.wasm.numThreads = 1;
        session ??= ort.InferenceSession.create("mobilenet.onnx", {executionProviders:["wasm"]})
            .catch(error => { session = undefined; throw error; });
        const encoder = await session;
        const started = performance.now();
        const result = await encoder.run({pixels: new ort.Tensor("float32", pixels, [1, 3, side, side])});
        if (request !== token) return;
        const vector = result.embedding.data;
        const norm = Math.hypot(...vector) || 1;
        const ranked = gallery.map(item => ({
            item, score: item.vector.reduce((total, x, i) => total + x * vector[i], 0) / norm
        })).sort((a, b) => b.score - a.score);
        const unique = new Map();
        for (const {item} of ranked) {
            if (!unique.has(item.id)) unique.set(item.id, item);
        }
        const products = [...unique.values()].slice(0, 4);
        get("results").replaceChildren(...products.map(item => card(item, item.id === knownId)));
        get("status").textContent = `${products.length} similar catalog products · encoding and search ${Math.round(performance.now() - started)} ms`;
    } catch (error) {
        if (request === token) get("status").textContent = "Search failed. Try another image or reload the demo.";
        console.error(error);
    }
}

fetch("catalog.json").then(response => {
    if (!response.ok) throw new Error("Catalog unavailable");
    return response.json();
}).then(({gallery, queries}) => {
    for (const item of queries) {
        const button = document.createElement("button");
        const image = new Image();
        image.src = item.image;
        image.alt = "";
        button.append(image, item.label);
        button.onclick = () => search(item.image, item.id, gallery);
        get("samples").append(button);
    }
    get("upload").onchange = async event => {
        const file = event.target.files[0];
        if (!file) return;
        if (file.size > 8_000_000) {
            ++token;
            get("status").textContent = "Choose an image smaller than 8 MB.";
            return;
        }
        const source = URL.createObjectURL(file);
        await search(source, null, gallery);
        // The query preview may keep the already decoded image.
        URL.revokeObjectURL(source);
    };
    get("status").textContent = "Select a sample or upload an image.";
}).catch(() => { get("status").textContent = "Could not load the catalog."; });
