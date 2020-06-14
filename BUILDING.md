# FAQ

# What is the workflow for developing new slides?

0. This is not needed, you can reload the browser manually to get updates.

1. Comment or uncomment katex in `index.html` and change the title.

2. Install `browser-sync` with and start while wathcing all files

```bash
npm install -g brower-sync
browser-sync start --server -w
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
decktape "http://localhost:3000/index.html" slides.pdf
```

## How to host on github pages?

1. Go to settings.
2. Enable GitHub Pages.
