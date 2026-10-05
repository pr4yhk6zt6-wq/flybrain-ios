# The world

Everything in this directory except `three.min.js` is **generated**. Run

```
python3 tools/step4_world.py     # makes fly.bin, frames.bin, world.json
python3 tools/serve_world.py     # serves this directory on :8000
```

| file | what it is |
|---|---|
| `index.html` | the viewer: three.js, a z-buffer of poses, and a HUD |
| `three.min.js` | vendored, r149, MIT — so the viewer needs no build step and no network |
| `fly.bin` | the animal's visible geometry, welded and packed once (6.3 MB) |
| `frames.bin` | one row of position + orientation per part per frame |
| `world.json` | what the parts are, what the animal did, and the parameters |

The `.bin` and `world.json` files are in `.gitignore`: they change on every
run and CI uploads them as the `step4-world` artifact instead.
