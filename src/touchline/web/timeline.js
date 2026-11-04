'use strict';
// Pure, provider-neutral timeline projection. All times are video-relative seconds.
(function (root) {
  function validTime(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
  function asArray(value) { return Array.isArray(value) ? value : []; }
  function entry(source, raw, index, sessionId) {
    if (!raw || typeof raw !== 'object') return null;
    const start = raw.video_start_s ?? raw.timestamp_s;
    const end = raw.video_end_s ?? start;
    if (!validTime(start) || !validTime(end) || end < start) return null;
    const text = String(raw.text ?? raw.label ?? raw.note ?? '').trim();
    if (!text) return null;
    const id = String(raw.transcript_id ?? raw.cue_id ?? raw.marker_id ?? raw.event_id ?? raw.observation_id ?? `${source}:${index}`);
    return {
      id, session_id: String(raw.session_id ?? sessionId), source,
      start_s: start, end_s: end, text,
      speaker_label: raw.speaker_label ?? raw.speaker ?? null,
      audio_recording_id: raw.audio_recording_id ?? null,
      player_id: raw.confirmed_player_id ?? null,
      local_track_id: raw.local_track_id ?? null,
      provider: raw.provider ?? null, model: raw.model ?? null,
      processing_status: raw.processing_status ?? raw.status ?? (source === 'transcript' ? 'completed' : null),
      coach_correction_status: raw.coach_correction_status ?? null,
      evidence_refs: asArray(raw.evidence_refs),
      local_only: source === 'manual_marker'
    };
  }
  function project(sessionId, data = {}, localMarkers = []) {
    const groups = [
      ['transcript', data.transcript?.segments],
      ['coach_cue', data.transcript?.manual_cues],
      ['manual_marker', localMarkers],
      ['detected_event', data.events],
      ['ai_observation', data.ai_observations]
    ];
    const output = [];
    for (const [source, items] of groups) {
      asArray(items).forEach((raw, index) => { const item = entry(source, raw, index, sessionId); if (item && item.session_id === String(sessionId)) output.push(item); });
    }
    return output.sort((a,b)=>a.start_s-b.start_s || a.end_s-b.end_s || a.source.localeCompare(b.source) || a.id.localeCompare(b.id));
  }
  function audioToVideo(audioOffsetS, videoAnchorS, driftSlope = 1) {
    if (![audioOffsetS, videoAnchorS, driftSlope].every(x=>typeof x === 'number' && Number.isFinite(x)) || audioOffsetS < 0 || videoAnchorS < 0 || driftSlope <= 0) throw new TypeError('Invalid media timing');
    return videoAnchorS + audioOffsetS * driftSlope;
  }
  const api = Object.freeze({project, audioToVideo, validTime});
  root.TouchlineTimeline = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
