# Recording-to-coach integration status — 4 November 2026

This isolated worktree starts at published `feature/player-mapping-modal-integration`
commit `9e4654f31d19793353d5240a1efacd5460ba0eee`. It selectively ports the
current coach frontend from `feature/frontend-unified-workspace`; the original
checkout and its staged index are untouched.

## Working local boundary

- The coach UI, explicit synthetic QA mode, strict production adapter, planned
  capability labels and disabled identity decisions are present.
- Local Studio serves historical runs, paged canonical observations and a
  readiness response. `POST /api/analyze` returns 503 and closes the connection;
  it neither accepts video bytes nor starts a worker. This prevents the older
  synchronous local detector path from being mistaken for full-recording Modal.
- An operator can import a **previously completed** panorama Sandbox export with
  `tools/import_panorama_export.py`. The command verifies an independently
  recorded original-video SHA-256, normalized-video and frame-map hashes, the
  exported overlay hash, the normalization record binding original to output,
  matching completed receipt/summary, exact frame-map coverage and observation
  shape. Detector confidence comes only from an exact frame-and-box match in
  the recorded raw-proposal stream; unmatched anchors fail import. It creates
  a fresh local run, publishes the
  manifest last, keeps motion predictions separate from measured detector
  anchors, and leaves every player identity unconfirmed. Existing run directories
  are never replaced. It does not decode media or execute a model.
- Local Studio verifies the imported overlay hash and frame-map hash before
  advertising the annotated URL. The coach can switch between original and
  annotated playback; source timeline positions use the exact frame map.
  Canonical source-pixel boxes are drawn only over the original video; the
  annotated export already contains provisional labels.

## Verified saved export from the 4 November transfer

The two supplied ZIPs were extracted together under
`/private/tmp/touchline-transfer-20261104/`. Their package verifier passed all
527 code and 361 evidence payload files. The completed four-minute ResNet export
was imported offline into
`/private/tmp/touchline-real-export-review-runs-v2/e61c5d3b-a2ca-4344-98cf-84cc186fd61e/`.
It yielded 22,563 canonical detector-anchor observations and 73,033 separate
motion/camera predictions. The original, normalized, frame-map and annotated
hashes matched the saved metadata and receipt. The run was opened through local
Studio HTTP: the annotated MP4 served with byte ranges, observation pagination
returned rows, and Chrome loaded both annotated and original video with no page
or network errors. The identity decision controls remained disabled. This is
review of an already completed operator export, not a new worker execution or
an authenticated backend acceptance run. After restarting Studio, its file-backed
GET still returned the completed run, 22,563 observations and the verified
annotated artifact. Both the transfer and local run remain
in temporary storage.

## Reconciled backend branch at `0d297fffc630fab9615cbfa201c1ba6c421eec4b`

The claimed routes are in `tools/fast_tracker/app.py`, not in the selected
Panorama controller. `POST /api/track` accepts an uploaded file or a caller
supplied local `video_path`, optional `weights` and `max_frames`; it starts a
thread running `process_tracking`. That thread uses `app_modal.run()` and
`track_video_gpu.remote()` or chunked functions in app
`soccertrack-v2-web-gpu`, with RT-DETR/YOLO defaults. It does **not** invoke
`tools/run_modal_panorama_review.py` or the reviewed ResNet Sandbox worker.
Jobs live in process memory, so restart recovery is absent. Results are
annotated/minimap MP4 and MOT-style CSV without the Panorama frame map, and
there is no proven canonical observation transformation. The route has no
authentication or authorization, accepts arbitrary server-local video paths
and model names, and does not bound the upload body. The correction handler
copies a manifest into a timestamp-derived child directory without validating
reviewer/roster identity or making run creation collision-safe; the child does
not contain a verified independent artifact bundle. These routes must remain
disconnected from production mode pending a reviewed backend replacement.

## Not integrated

- No authenticated recording upload, object storage, idempotent asynchronous job
  API, attempt persistence, authorized artifact URLs or process-restart recovery
  is implemented by this worktree. The existing panorama controller is a
  single-use, fixed-input operator script, not a reusable backend dispatcher.
- No Modal execution, deployment, recording upload, credential read or provider
  call was performed here. A mock worker cannot prove the real Sandbox path.
- No authenticated roster or immutable child-run correction API exists here;
  confirm/reject remains disabled. Browser-local development drafts remain
  labelled as such. Tracking quality remains provisional.
- The importer is an operator-only local bridge, not production persistence.
  Studio's unauthenticated local artifact routes must not be exposed publicly.

## Next reviewed contract

The backend contract still needs authenticated upload/job/attempt,
durable storage, artifact, roster and correction schemas with authorization and idempotency.
The worker integration still needs a reusable bounded Sandbox input/output contract that
accepts an authorized recording and approved checkpoint, plus a reviewed adapter
for experimental ResNet observations and exact frame-map provenance. Then the
production route map in `src/touchline/web/api-client.js` can be connected and
one separately authorized real recording run can test the complete path.

Until then, keep production upload and correction controls disabled. Do not
promote provisional track IDs to named players or claim speed, distance, tracking
accuracy, improvement, or approved reports from this local bridge.
