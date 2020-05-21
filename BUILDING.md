# FAQ

# What is the workflow for developing new slides?

1. Install `browser-sync` with

```bash
npm install -g brower-sync
```

2. Start `brower-sync`

```
browser-sync start --server --files "*.*" --startPath "?p=slides.md"
```

or

```
browser-sync start --server --files "*.*"
```

## How to save as pdf

1. Install `decktape` with

```bash
npm install -g decktape
```

2. Make sure `brower-sync` or another http server method is used:

```bash
python -m http.server 8000
```

3. Use `decktape` to save pdf.

```bash
decktape "http://localhost:3000/index.html?p=slides.md" slides.pdf
```

## How to host on github pages?

1. Go to settings.
2. Enable GitHub Pages.
