fetch("comparison.json").then(response => {
    if (!response.ok) throw new Error("Comparison unavailable");
    return response.json();
}).then(({query, methods}) => {
    const element = id => document.getElementById(id);
    const image = (src, label) => {
        const img = new Image();
        img.src = src;
        img.alt = label;
        img.width = 92;
        img.height = 92;
        return img;
    };
    const caption = document.createElement("p");
    caption.textContent = `${query.label} · ${query.condition}`;
    element("compare-query").append(image(query.image, query.label), caption);
    for (const method of methods) {
        const column = document.createElement("div");
        const title = document.createElement("h3");
        title.textContent = method.name;
        const grid = document.createElement("div");
        grid.className = "grid";
        method.matches.forEach((match, rank) => {
            const figure = document.createElement("figure");
            figure.className = match.correct ? "hit" : rank === 0 ? "wrong" : "";
            const title = document.createElement("figcaption");
            title.textContent = `${rank+1}. ${match.label}`;
            if (match.correct || rank === 0) {
                const label = document.createElement("small");
                label.textContent = match.correct ? "Same product" : "Wrong top match";
                title.append(document.createElement("br"), label);
            }
            figure.append(image(match.image, match.label), title);
            grid.append(figure);
        });
        column.append(title, grid);
        element("compare-models").append(column);
    }
    element("compare-note").textContent = "Recorded image rankings. Each product has multiple views, so it can appear more than once here.";
}).catch(() => {
    document.getElementById("compare-note").textContent = "Could not load the recorded rankings.";
});
