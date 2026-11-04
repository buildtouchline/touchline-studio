# Touchline Studio — Frontend Webapp Guide

Welcome to the frontend webapp workspace! This directory contains the complete client-side application for Touchline Review Studio.

## 📁 Directory Structure

```
src/touchline/web/
├── index.html     # Semantic HTML layout (Video stage, Scrubber, Mic bar, Tabs, Modals)
├── styles.css     # Design system, CSS variables, dark theme, responsive grid, animations
├── app.js         # Client logic (HTML5 video playback, canvas overlay, audio recorder, API client)
└── README.md      # API contract & developer instructions
```

---

## ⚡ Live Development Workflow

1. Start the server (runs from repo root):
   ```bash
   python -m touchline studio --port 8765
   ```
2. Open your browser at `http://127.0.0.1:8765/`.
3. **No rebuild step needed:** Whenever you edit `styles.css`, `app.js`, or `index.html`, simply press `Ctrl + F5` (or `Shift + Reload`) in your browser to see your changes immediately.

---

## 🔌 Backend REST API Contract

The Python backend provides these endpoints. You can call them using standard `fetch()`:

### 1. Match Footage & Sessions
* **`GET /api/runs`**
  * Returns list of available match sessions.
  * Response:
    ```json
    [
      {
        "id": "calibrated-snmot097",
        "name": "SNMOT-097 Match Video (Calibrated Speed m/s)",
        "calibrated": true,
        "total_tracks": 27,
        "duration_s": 1.96,
        "has_video": true,
        "has_transcript": true
      }
    ]
    ```

* **`GET /api/run/<session_id>`**
  * Returns complete tracking telemetry, player roster, speed, distance, and subtitle transcript.
  * Response keys:
    * `sessionData.video_url`: URL to stream video (e.g. `/runs/calibrated-snmot097/source_video.mp4`).
    * `sessionData.observations`: Array of per-frame detections with `bbox_xyxy: [x1, y1, x2, y2]`.
    * `sessionData.analytics.player_roster`: Array of player workload objects (speeds, ground trajectories, distances).
    * `sessionData.transcript.segments`: Array of speech turns (`text`, `video_start_s`, `video_end_s`).

### 2. Video Streaming (HTTP 206 Partial Content)
* **`GET /runs/<session_id>/source_video.mp4`**
  * Hardware-accelerated byte-range stream. Bind directly to `<video src="...">`.

### 3. Bluetooth Collar Mic & Sarvam AI Voice Transcription
* **`POST /api/audio/transcribe?run_id=<session_id>&video_anchor_start_s=<seconds>&monotonic_start_ms=<ms>&language_code=<lang>`**
  * Body: Raw audio binary (`application/octet-stream`) from browser `MediaRecorder` (`audio/webm` or `audio/wav`).
  * Supported languages: `en-IN`, `hi-IN`, `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`.
  * Response: Synchronized `touchline.transcript.v1` object with exact video timestamp anchors.

### 4. Custom Tactical Cues / Subtitles
* **`POST /api/run/<session_id>/transcript`**
  * Body: Updated transcript JSON object (`segments` and `manual_cues`).
  * Response: `{"status": "ok", "transcript": {...}}`.

### 5. Video & Subtitle Exports
* **`GET /api/run/<session_id>/subtitles.srt`**: SubRip `.srt` subtitle file download.
* **`GET /api/run/<session_id>/subtitles.vtt`**: WebVTT `.vtt` file download.
* **`GET /api/run/<session_id>/video_with_subtitles.mp4`**: Renders & downloads MP4 with burned-in subtitles.

---

## 🎨 Key UI Components & IDs in `index.html`

* `#matchVideo`: The native HTML5 video player.
* `#overlayCanvas`: Canvas sitting directly over the video (`1920 × 1080` coordinate space).
* `#videoSubtitlesOverlay`: On-screen subtitle badge for live playback speech.
* `#videoSlider`: Scrubber timeline slider.
* `#btnPlayPause`, `#btnStepBack`, `#btnStepFwd`: Playback transport buttons.
* `#btnRecordMic`: Collar mic recording trigger.
* `#subtitlesList`: Sidebar feed for commentary cards.
* `#playersList`: Sidebar feed for tracked squad roster.
* `#exportDropdown`: Top menu for `.mp4`, `.srt`, `.vtt` downloads.
