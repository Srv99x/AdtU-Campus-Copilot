# assets/

`index.html` references `assets/adtu-logo.png` for the sidebar brand mark.

**That file does not exist yet.** No ADTU logo asset was found anywhere in
this repository during this task (searched exhaustively — only evaluation
confusion-matrix PNGs and a gate-calibration histogram exist, no logo). Per
instructions, no logo was invented or redrawn.

The page degrades gracefully in the meantime: `index.html`'s inline
`<script>` listens for the `<img>`'s `error` event and falls back to a text
wordmark ("AdtU" / "Campus Copilot") instead of showing a broken-image icon.

**To finish this:** drop the real ADTU logo as `adtu-logo.png` in this
directory. No code change is needed — the `<img>` tag will pick it up
automatically and the text fallback will stop showing.
