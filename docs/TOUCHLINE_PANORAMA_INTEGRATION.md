# Touchline Panorama/ResNet recording-to-coach integration

**Review proposal, 4 November 2026.** This document describes the intended
interface between the existing coach workspace and the published Panorama
Sandbox worker. Proposed HTTP routes below are **not live endpoints**. The
completed four-minute export is private review evidence, not an accepted
tracking model or a successful browser-to-Modal acceptance test.

## Source revisions and present state

| Component | Pin | Verified state |
| --- | --- | --- |
| Coach frontend | `feature/frontend-unified-workspace` at `e2386c93ed77801a9a5489f2a6e19a444774135d`, selectively ported onto this branch | Mode-aware client; explicit synthetic QA; local review; production fails closed when routes are unconfirmed. Browser-local development drafts are not server records. |
| Target worker | `feature/player-mapping-modal-integration` at `9e4654f31d19793353d5240a1efacd5460ba0eee` | `tools/run_modal_panorama_review.py:run` creates a bounded A10 Modal Sandbox and executes `tools/modal_panorama_worker.py:main`. It uses fixed operator paths, hashes and a consumed allowance; it is not a reusable upload/job service. |
| Other backend | `feature/player-mapping-voiceover` commit `0d297fffc630fab9615cbfa201c1ba6c421eec4b`, inspected separately | `tools/fast_tracker/app.py:do_POST` exposes `/api/track`, but `process_tracking` invokes `track_video_gpu.remote()` or chunk functions in a different RT-DETR/YOLO app. Its job dictionary is memory-only. Do not connect it as Panorama. The remote branch has moved since this inspected pin. |
| Saved export | Run `e61c5d3b-a2ca-4344-98cf-84cc186fd61e` in the private 4 November transfer | Package payloads and original/normalized/frame-map/weight/overlay hashes verified. Offline importer yielded 22,563 canonical detector anchors and 73,033 separate motion/camera predictions. Local Studio reopened the run after restart and played original/annotated media. No fresh worker ran. |

The historical baseline reports **84.29% detection AP50**, **93.27% detection
recall**, **60.26% tracked-box recall**, **31.27% provisional global identity
F1**, and **224 identity changes** on its defined sampled comparison. Detection
metrics are not player identity accuracy. Tracking quality and coach identity
associations remain unaccepted. No physical distance or speed is validated.

## Agreed architecture

```text
Existing coach frontend: select/upload, job state, Analysis, timeline, track review
  -> authenticated backend: authorized upload, durable job and attempt, correction
  -> bounded Modal Sandbox: the Panorama/ResNet controller and worker
  -> verified export before termination: overlay, observations, predictions, map
  -> durable run store: source/normalized/model/code hashes, manifest, artifacts
  -> authorized backend retrieval: range video, paged evidence, child-run history
  -> existing coach workspace: review provisional tracks and saved coach decisions
```

The browser receives neither Modal credentials nor worker filesystem paths. The
backend chooses the approved checkpoint, code revision and resource ceiling; it
must not accept arbitrary paths, scripts, URLs or model names from a request.
One job may have multiple attempts; job, attempt, run and idempotency IDs must
remain distinct. A successful status requires exported, hash-verified artifacts
in durable storage. Failed, interrupted and partial attempts stay visible.

The existing operator controller stages `video24.mp4`, `weights.pth`,
`frame-map.jsonl`, `code.zip`, the worker script and `panorama-job.json` under
`/work`. It polls structured worker state, validates the 5,760-frame historical
summary, copies `overlay24.mp4` and `evidence.zip` locally, then terminates the
Sandbox. The historical limits were A10, four CPUs, 16 GiB, 2,700 seconds and a
recorded $1.80 reservation; billing remains unreconciled. These consumed limits
are **not** permission or a reusable command for a new recording.

The worker's raw row has `frame_index`, normalized `timestamp_s`, `track_id`,
`bbox_xyxy`, `detector_bbox_xyxy` and an `evidence` class. The exact detector
score comes from `raw_proposals.jsonl` joined by normalized frame and exact
detector box. Canonical `touchline.observation.v1` requires nested `frame`,
`local_track_id`, `detector_score`, `association_state`, and nullable `player_id`.
Motion/camera predictions stay outside canonical detector-anchor observations.
The original 25 FPS media time comes from the 5,760-row source-to-normalized
frame map, never from an assumed FPS ratio. The importer also requires the
normalization record to bind the original hash, normalized hash and map hash.

## Proposed backend contract for integration review

These names and shapes are **proposals**, not an assertion that the routes exist.
Authentication and authorization apply to every operation and artifact. Errors
should have a stable typed code and safe message. No response includes a server
filesystem path, provider credential, invented percentage or ETA.

| Purpose | Proposed route and minimum contract |
| --- | --- |
| Create/upload recording | `POST /api/runs` with authorized media bytes or finalized upload reference, session/team reference, and idempotency key. Return `202 {job_id, run_id, status:"queued", source_sha256}` only after durable admission. Large upload may use separately agreed chunk/finalize routes. |
| Job status | `GET /api/jobs/{job_id}` returns distinct `job_id`, `attempt_id`, `run_id`, `status` (`queued`, `running`, `partial`, `succeeded`, `failed`, `cancelled`), latest stage and persisted progress events. No fabricated percentages. |
| List/get runs | `GET /api/runs`, `GET /api/runs/{run_id}` return authorized run summaries, parent/child lineage, status and verified artifact availability. |
| Manifest | `GET /api/runs/{run_id}/manifest` returns immutable source, normalized, frame-map, code, checkpoint and artifact hashes plus coverage and failure/partial reason. |
| Observations | `GET /api/runs/{run_id}/observations?offset=&limit=` returns canonical `touchline.observation.v1` rows and bounded paging cursor; predictions have a separate typed resource. |
| Artifacts | Run-specific authorized URL, e.g. `GET /api/runs/{run_id}/artifacts/{kind}`, with verified content type, HTTP range support for MP4, and no arbitrary path parameter. |
| Coach correction | `POST /api/runs/{run_id}/corrections` accepts a reviewed local track, a roster player ID from authorized data, decision, reviewer from authentication, note and exact evidence/run version. Return a new immutable `child_run_id` only after atomic persistence. Reject concurrent conflicting revisions. |

The frontend adapter must normalize these responses in one place and show
loading, queued, running, partial, failed and empty states. Production mode must
never fall back to fixtures. The separately inspected backend's current `/api/runs/...` file-backed ideas
may inform implementation, but the inspected route set lacks authentication,
safe upload bounds, durable job state, canonical observations and safe immutable
corrections; its `/api/track` selects a different model path. This branch does
not silently enable those routes.

## Next phase and acceptance gates

1. **Not implemented:** review a parameterized, reusable adapter around the
   actual Panorama Sandbox controller; approve input/checkpoint/code pins, fresh
   allowance, runtime bounds and export-before-termination behavior. A review of
   the worker contract is needed before any fresh job.
2. **Not implemented:** authenticated upload, idempotent durable job/attempt
   records and authorized source storage. Backend maintainers could review
   or contribute the run-management and API design; no assignment is implied.
3. **Partially implemented locally:** offline import of a completed export,
   canonical detector-anchor conversion, exact frame-map preservation, local
   manifest and paged Studio observations. Move this logic behind verified
   backend storage and typed failure handling.
4. **Not implemented:** authorized run/manifest/observation/artifact retrieval,
   range video and restart-safe job progress. Connect the existing frontend
   adapter only after the HTTP schema is reviewed and tested.
5. **Not implemented:** authenticated roster identity decision and immutable
   child-run correction with reopen/revision tests. Local track IDs remain
   provisional; browser-local drafts remain labelled.
6. **Requires separate real Modal validation:** mock the complete lifecycle,
   corrupt/missing artifacts, duplicate requests, failure/partial states and
   restart first. Then obtain explicit approval for one fresh bounded recording
   run, verify exact hashes and costs, open it in the coach workspace, save a
   reviewed correction, and reopen both parent and child after backend restart.

The current UI keeps tactical minimap, MOTChallenge export, jersey recognition,
validated speed/distance, automatic retraining, coach voice commentary and
advanced scoring visibly **Planned** without fabricated outputs.

## Local verification and transfer limits

[`INTEGRATION_SPRINT_STATUS.md`](INTEGRATION_SPRINT_STATUS.md) records the offline
tests and saved-export browser check. The supplied ZIPs and imported run remain
under `/private/tmp`; they are private, temporary and excluded from this PR.
The coach workspace is reviewable against the retained local export, but this
does not prove new upload, production authentication, durable cloud storage,
accepted tracking quality, or a completed browser-to-Modal job.
