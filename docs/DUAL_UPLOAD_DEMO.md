# Separate video/audio demo implementation guide

31 October 2026. Planning only; the durable session/API/dual-upload workflow is not implemented yet. Existing audio normalization and timing helpers are prerequisites, not a complete verified synchronization system. In particular, a current helper clamps negative mapped times; TL-045 requires preserving out-of-range cues explicitly before reuse in the new workflow.

Use two source slots per private session: silent video and separately recorded coach audio. Keep original bytes/hashes, source PTS/time base, audio sample clocks and metadata provenance. Prefer matched events/sidecars, support manual offset review, fit drift only with anchors, and expose metadata-only estimates. File modification/creation metadata alone does not prove capture time. One recording per slot is the controlled demo; pauses/restarts require explicit segmented mapping. Keep transcript and alignment revisions separate and invalidate stale reports after corrections.

Core target: restricted upload → automatic supported measurements/draft score → synchronized coach playback/correction → recompute improvement plan → exact approval/export. Manual timestamped transcript import/cues is the ASR fallback, not a claim of automatic transcription. Unverified player identity or calibration leaves metrics unscored. Full continuous training, autonomous repair/promotion and native Android delivery are follow-up.

Modal CPU is the proposed actual web host; Modal GPU/Volumes handle analysis and reviewed training artifacts. Meta-hosted sam-3.1 assists segmentation/teacher review. Baseten proposes bounded repairs behind [the rulebook](AUTOFIX_RULEBOOK.md). Fireworks is optional model inference, not documented frontend hosting. No provider deployment or paid job occurred in this planning update.

See [pinned tickets and complete HLD/20-hour schedule](TICKET_REFERENCE.md) and [secret/endpoint names](MODAL_PROVIDER_CONTRACT.md). All new code remains on feature/modal-training; existing PRs are unchanged. Agent access to actual .env/credentials remains forbidden.

## Analysis-first update

Read [ANALYSIS_FIRST_RUNBOOK.md](ANALYSIS_FIRST_RUNBOOK.md) for strict 1080p24 preflight, proposed 95% acceptance, fixed $50/$75/$50/$50USD project ceilings, provider dashboard steps and operator-only ENV names. Local guards are implemented; paid provider wiring and player-scoring runtime remain open. Earlier training-first/zero-allocation-only scheduling is superseded; no unverified job is enabled.
