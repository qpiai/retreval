# Demo video (Remotion)

Composites a branded **intro + outro** around the Playwright-captured app
footage and renders the README's `docs/demo.mp4` / `docs/demo.gif`.

```
intro (4s)  →  real app walkthrough  →  outro (6s)
🌳 brand        Plan → Expand → Refine        stats + CTA + links
                → Score → Validate
```

## Build it

From the repo root, with both servers up (static UI on `:7575`, mock backend on
`:7373`):

```bash
scripts/build_demo.sh
# 1. capture 1080p footage  → video/public/app.mp4
# 2. remotion render        → docs/demo.mp4
# 3. ffmpeg + gifsicle      → docs/demo.gif
```

Or step by step:

```bash
python scripts/capture_demo.py        # → video/public/app.mp4
npm --prefix video install            # first time only
npm --prefix video run render         # → docs/demo.mp4
npm --prefix video run studio         # live-edit the intro/outro
```

## Layout

| file | role |
|---|---|
| `src/index.ts` | Remotion entry (`registerRoot`). |
| `src/Root.tsx` | `<Composition>`; sizes the timeline to intro + footage + outro via `calculateMetadata`. |
| `src/Demo.tsx` | The intro, the `OffthreadVideo` app clip (sped 1.4×), the outro, and the `<Audio>` BGM bed. |
| `public/app.mp4` | Captured footage (gitignored — regenerate with the capture script). |
| `public/bgm3.mp3` | Background music (gitignored — copied from `docs/bgm3.mp3` by `build_demo.sh`). |

## Music

The track lives at `docs/bgm3.mp3`; `build_demo.sh` copies it to
`public/bgm3.mp3` and `Demo.tsx` mixes it under the video at ~32 % with a fade
in/out. Swap it by replacing `docs/bgm3.mp3` (and the `BGM` constant in
`Demo.tsx` if you rename it). If the file is missing, the build generates a
silent placeholder so the render still succeeds. Edit `Demo.tsx` to restyle the
brand cards, stats, or CTA.
