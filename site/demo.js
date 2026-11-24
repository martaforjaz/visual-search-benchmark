const byId = id => document.getElementById(id);
const {rows, samples} = window.report;
let selected = 0;

function tile(src, label, mismatch = false) {
    const element = document.createElement("div");
    element.className = "tile";
    const image = document.createElement("img");
    image.src = src;
    image.alt = label;
    const caption = document.createElement(mismatch ? "small" : "span");
    caption.textContent = label;
    element.append(image, caption);
    return element;
}
function show(matches, label) {
    byId("matches").replaceChildren(...matches.map(item => tile(item.image, item.label, label && label !== item.label)));
    byId("match-summary").textContent = label
        ? `Query: ${label} · ${matches.filter(item => item.label === label).length}/10 same-class results`
        : "Uploaded query · exact search";
}
function refresh() {
    const sample = samples[selected];
    [...byId("examples").children].forEach((button, i) => button.setAttribute("aria-pressed", String(i === selected)));
    show(sample[byId("method").value], sample.label);
}
samples.forEach((sample, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.append(tile(sample.query, i === 1 ? sample.label + " · hard case" : sample.label));
    button.addEventListener("click", () => { selected = i; refresh(); });
    byId("examples").append(button);
});
function metrics() {
    const row = rows.find(value => value.size === +byId("size").value && value.candidates === +byId("count").value);
    byId("metrics").textContent = `Recall@10 ${(row.recall*100).toFixed(1)}% · Same-class Precision@10 ${(row.precision*100).toFixed(1)}% · p95 ${row.p95_ms} ms · ${(row.index_bytes/1048576).toFixed(2)} MiB stored`;
}
if (location.hostname === "127.0.0.1" && location.port === "8777") {
    byId("upload").hidden = false;
    byId("upload").onsubmit = async event => {
        event.preventDefault();
        byId("match-summary").textContent = "Searching…";
        try {
            const response = await fetch("/search", {method:"POST", body:byId("image").files[0]});
            if (!response.ok) throw Error();
            show(await response.json());
        } catch {
            byId("match-summary").textContent = "Could not search. Choose a valid image under 2 MB.";
        }
    };
}
byId("method").onchange = refresh;
byId("size").onchange = byId("count").onchange = metrics;
refresh();
metrics();
