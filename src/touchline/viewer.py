"""Interactive Level 1 football inspection viewer and analytics dashboard.

Produces a standalone, zero-dependency HTML5 application with:
- Frame-by-frame and continuous playback with scrubber timeline
- Player bounding boxes, track labels, target indicators, and ground contact points
- Interactive 2D spatial density heatmap toggle
- Player analytics roster cards (distance, speed profile, visible time, confirmed identity)
- Explicit player association and target selection tool (TL-004, TL-004.1)
- Level 1 coach attempt annotation and review tool (TL-005.1, TL-005.2)
- Explicit synthetic and experimental warning banners
"""
import html
import json
from typing import Any


def render_inspection_dashboard(
    manifest: dict[str, Any],
    analytics: dict[str, Any],
    observations: list[dict[str, Any]],
    transcript: dict[str, Any] | None = None,
) -> str:
    """Generate self-contained interactive HTML5 analytics and review dashboard."""
    is_synthetic = bool(manifest.get("synthetic", False))
    banner = (
        "SYNTHETIC CONTRACT FIXTURE — NOT FIELD EVIDENCE"
        if is_synthetic
        else "PRELIMINARY / UNVALIDATED TRACKING & WORKLOAD ESTIMATES — ROSTER IDENTITY UNCONFIRMED"
    )

    width = analytics["session"]["frame_width"]
    height = analytics["session"]["frame_height"]

    # Fallback static SVG trails for backwards compatibility
    trails = []
    for player in analytics.get("player_roster", []):
        tid = player["local_track_id"]
        pts = " ".join(f"{pt['ground_contact_px'][0]},{pt['ground_contact_px'][1]}" for pt in player["trajectory"])
        color = f"#{(tid * 2381393) % 0xffffff:06x}"
        trails.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    static_trails_svg = "".join(trails)

    # Embed data safely as JSON for the client-side viewer
    safe_analytics_json = json.dumps(analytics, allow_nan=False).replace("</script>", "<\\/script>")
    safe_observations_json = json.dumps(observations[:1000], allow_nan=False).replace("</script>", "<\\/script>")
    safe_transcript_json = json.dumps(transcript or {}, allow_nan=False).replace("</script>", "<\\/script>")

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Touchline Football Analytics — Run Inspection</title>
<style>
  :root {{
    --bg-main: #0b1319;
    --bg-card: #14212b;
    --bg-card-sub: #1c2d3b;
    --pitch-bg: #102d23;
    --pitch-line: rgba(255, 255, 255, 0.45);
    --accent: #00d084;
    --accent-target: #ff6b35;
    --accent-warn: #f5a623;
    --accent-danger: #ff4d4f;
    --text-primary: #f0f6f8;
    --text-secondary: #9bb0be;
    --border: #223847;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 16px;
  }}
  header {{
    max-width: 1400px;
    margin: 0 auto 16px;
  }}
  .banner {{
    padding: 12px 18px;
    background: #684d17;
    color: #fff;
    font-weight: 700;
    letter-spacing: 0.5px;
    border-radius: 6px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .banner.synthetic {{ background: #5c350b; border-left: 6px solid #e67e22; }}
  .header-meta {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
  }}
  h1 {{ font-size: 22px; font-weight: 600; color: var(--text-primary); }}
  .tag {{
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 4px;
    background: var(--bg-card-sub);
    color: var(--accent);
    font-family: monospace;
  }}
  .tag.uncalibrated {{ color: var(--accent-warn); background: #382914; }}
  .tag.target {{ color: #fff; background: var(--accent-target); }}
  .main-layout {{
    max-width: 1400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 20px;
  }}
  @media (max-width: 1024px) {{
    .main-layout {{ grid-template-columns: 1fr; }}
  }}
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  .card-title {{
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
  }}
  .viewport-container {{
    position: relative;
    width: 100%;
    aspect-ratio: {width} / {height};
    background: var(--pitch-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  canvas {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }}
  .timeline-bar {{
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .controls-row {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  button {{
    background: var(--bg-card-sub);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }}
  button:hover {{ background: #273d4f; border-color: var(--accent); }}
  button.active {{ background: var(--accent); color: #0b1319; border-color: var(--accent); }}
  .slider {{
    flex: 1;
    height: 6px;
    background: var(--bg-card-sub);
    outline: none;
    border-radius: 3px;
    cursor: pointer;
  }}
  .time-display {{
    font-family: monospace;
    font-size: 13px;
    color: var(--text-secondary);
    min-width: 140px;
    text-align: right;
  }}
  .toggles-row {{
    display: flex;
    gap: 16px;
    font-size: 13px;
    color: var(--text-secondary);
    align-items: center;
    margin-top: 6px;
  }}
  .toggles-row label {{ display: flex; align-items: center; gap: 6px; cursor: pointer; }}
  .roster-list {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 480px;
    overflow-y: auto;
    padding-right: 4px;
  }}
  .player-card {{
    background: var(--bg-card-sub);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }}
  .player-card:hover {{ background: #223748; }}
  .player-card.selected {{
    border-color: var(--accent);
    background: #17343b;
  }}
  .player-card.is-target {{
    border-left: 4px solid var(--accent-target);
  }}
  .player-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }}
  .track-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    font-size: 14px;
  }}
  .color-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }}
  .quality-badge {{
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: 700;
  }}
  .quality-high {{ background: #0c3e2f; color: #00d084; }}
  .quality-moderate {{ background: #3d2f0d; color: #f5a623; }}
  .quality-contested {{ background: #4a1313; color: #ff4d4f; }}
  .target-badge {{ background: #4e1d09; color: #ff6b35; border: 1px solid #ff6b35; }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    font-size: 12px;
  }}
  .metric-item {{
    background: rgba(0, 0, 0, 0.2);
    padding: 6px 8px;
    border-radius: 4px;
  }}
  .metric-val {{
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
  }}
  .metric-label {{
    color: var(--text-secondary);
    font-size: 11px;
  }}
  .review-box {{
    margin-top: 8px;
  }}
  .review-select {{
    width: 100%;
    background: var(--bg-card-sub);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 6px;
    border-radius: 4px;
    margin-bottom: 8px;
    font-size: 13px;
  }}
  textarea {{
    width: 100%;
    background: var(--bg-card-sub);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 6px;
    border-radius: 4px;
    font-size: 12px;
    resize: vertical;
    min-height: 50px;
  }}
  details {{
    margin-top: 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
  }}
  summary {{ cursor: pointer; font-weight: 600; color: var(--text-secondary); }}
  pre {{
    margin-top: 8px;
    padding: 12px;
    background: #060b0e;
    border-radius: 4px;
    font-size: 12px;
    overflow-x: auto;
    color: #a0c2d3;
  }}
  .svg-trail-container {{ display: none; }}
</style>
</head>
<body>
<header>
  <div class="banner { 'synthetic' if is_synthetic else '' }">{banner}</div>
  <div class="header-meta">
    <h1>Touchline Match Inspection</h1>
    <div>
      <span class="tag">Run: {manifest.get("run_id", "local")[:8]}</span>
      <span class="tag">Frames: {analytics["session"]["selected_frames"]}/{analytics["session"]["decoded_frames"]}</span>
      <span class="tag">Duration: {analytics["session"]["duration_s"]}s</span>
      <span class="tag { 'uncalibrated' if not analytics['calibration']['calibrated'] else '' }">
        { 'Calibrated (m)' if analytics['calibration']['calibrated'] else 'Uncalibrated (px)' }
      </span>
      { '<span class="tag target">🎯 Target Participant Tracked</span>' if analytics['quality_indicators'].get('target_tracks_count', 0) > 0 else '' }
    </div>
  </div>
</header>

<div class="main-layout">
  <!-- Viewport Section -->
  <div>
    <div class="card">
      <div class="card-title">
        <span>Pitch Visualisation & Tracking</span>
        <span id="activePlayerLabel">All Players</span>
      </div>
      <div class="viewport-container" id="viewport">
        <video id="sourceVideo" src="{html.escape(str(manifest.get('video_file', 'source_video.mp4')))}" muted playsinline preload="auto" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:fill;display:none;"></video>
        <canvas id="pitchCanvas" width="{width}" height="{height}"></canvas>
      </div>

      <div class="timeline-bar">
        <div class="controls-row">
          <button id="playBtn">▶ Play</button>
          <button id="stepBackBtn">◀ Step</button>
          <button id="stepFwdBtn">Step ▶</button>
          <input type="range" id="frameSlider" class="slider" min="0" max="0" value="0">
          <div class="time-display" id="timeDisplay">Frame 0 / 0 (0.00s)</div>
        </div>

        <div style="margin-top:4px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-secondary);margin-bottom:3px;">
            <span>Visibility & Coverage Strip (Image-Space Exposure)</span>
            <span>🟩 Observed &nbsp; ⬛ Missing/Gap (Not Inactivity)</span>
          </div>
          <div id="visibilityStrip" style="position:relative;height:10px;background:#263238;border-radius:4px;overflow:hidden;border:1px solid var(--border);"></div>
        </div>
        <div class="toggles-row">
          <label><input type="checkbox" id="toggleVideo" checked> Source Video</label>
          <label><input type="checkbox" id="toggleBoxes" checked> Bounding Boxes</label>
          <label><input type="checkbox" id="toggleTrails" checked> Motion Trails</label>
          <label><input type="checkbox" id="toggleContacts" checked> Ground Contacts</label>
          <label><input type="checkbox" id="toggleHeatmap"> Spatial Heatmap</label>
          <label><input type="checkbox" id="toggleLoop" checked> Loop Playback</label>
        </div>
      </div>
    </div>

    <!-- Coach Bluetooth Collar Mic & Sarvam AI Transcript Timeline -->
    <div class="card" id="coachMicCard">
      <div class="card-title">
        <span>🎙️ Coach Bluetooth Collar Mic &amp; Sarvam AI Transcript Timeline</span>
        <span class="tag uncalibrated">Speaker Turn: Unverified</span>
      </div>
      <div style="display:grid;grid-template-columns:1.2fr 0.8fr auto auto auto;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
        <select id="micDeviceSelect" class="review-select" style="margin-bottom:0;">
          <option value="">🎧 Default Microphone / Bluetooth Collar Mic (Click Refresh after granting permission)</option>
        </select>
        <select id="sarvamLangSelect" class="review-select" style="margin-bottom:0;">
          <option value="en-IN" selected>Sarvam AI: English / Hinglish (en-IN)</option>
          <option value="hi-IN">Sarvam AI: Hindi (hi-IN)</option>
          <option value="ta-IN">Sarvam AI: Tamil (ta-IN)</option>
          <option value="te-IN">Sarvam AI: Telugu (te-IN)</option>
          <option value="kn-IN">Sarvam AI: Kannada (kn-IN)</option>
          <option value="ml-IN">Sarvam AI: Malayalam (ml-IN)</option>
        </select>
        <button type="button" id="refreshMicBtn" title="Refresh audio input devices">🔄 Devices</button>
        <button type="button" id="startMicBtn" style="background:#0c3e2f;border-color:#00d084;color:#00d084;">🔴 Record Continuous</button>
        <button type="button" id="stopMicBtn" disabled style="background:#4a1313;border-color:#ff4d4f;color:#ffb3b3;opacity:0.5;">⏹️ Stop</button>
        <button type="button" id="pttMicBtn" style="background:#1b384f;border-color:#1e90ff;color:#1e90ff;user-select:none;cursor:pointer;font-weight:bold;">🎙️ Hold to Speak (Push-to-Talk)</button>
      </div>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
        <input type="text" id="liveCueNoteInput" placeholder="Optional quick cue note while recording (e.g., 'Press trigger here', 'Scan before receiving')..." class="review-select" style="flex:1;margin-bottom:0;">
        <button type="button" id="markLiveCueBtn">📌 Stamp Video/Audio Cue Now</button>
        <label style="font-size:12px;color:var(--text-secondary);cursor:pointer;background:var(--bg-card-sub);padding:6px 10px;border-radius:4px;border:1px solid var(--border);">
          📂 Upload Audio (.wav/.webm/.mp3)
          <input type="file" id="uploadAudioFileInput" accept="audio/*" style="display:none;">
        </label>
      </div>
      <div id="micStatusBanner" style="font-size:12px;color:var(--text-secondary);padding:7px 10px;background:rgba(0,0,0,0.25);border-radius:4px;margin-bottom:8px;">
        Ready: Select your <strong>Bluetooth Collar Mic</strong>, play or scrub the match video above, and click <strong>🔴 Record Collar Mic</strong>. Both <code>video_start_s</code> (PTS-synced) and <code>audio_start_s</code> are recorded automatically.
      </div>
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-secondary);margin-bottom:3px;">
          <span>Coach Commentary &amp; Cue Markers on Video Timeline (Click any marker or row below to seek video + boxes)</span>
          <span>🟧 Sarvam Transcript Segment &nbsp; 🟦 Live Coach Cue</span>
        </div>
        <div id="transcriptTimelineStrip" style="position:relative;height:14px;background:#1a2730;border-radius:4px;overflow:hidden;border:1px solid var(--border);"></div>
      </div>
      <div id="transcriptSegmentsList" style="display:flex;flex-direction:column;gap:6px;max-height:220px;overflow-y:auto;"></div>
    </div>

    <!-- Static Preview for legacy check -->
    <div class="card svg-trail-container">
      <h3>Static Trail Render</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Sampled image-space trails">{static_trails_svg}</svg>
    </div>

    <!-- Match Diagnostics & Limitations -->
    <div class="card">
      <div class="card-title">Match Quality & Association Lineage (TL-004)</div>
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-val">{analytics["quality_indicators"]["total_tracks"]}</div>
          <div class="metric-label">Active Track IDs</div>
        </div>
        <div class="metric-item">
          <div class="metric-val">{analytics["quality_indicators"].get("confirmed_players_count", 0)}</div>
          <div class="metric-label">Confirmed Player Associations</div>
        </div>
        <div class="metric-item">
          <div class="metric-val">{analytics["quality_indicators"].get("target_tracks_count", 0)}</div>
          <div class="metric-label">Target Participant Tracks</div>
        </div>
        <div class="metric-item">
          <div class="metric-val">{analytics["quality_indicators"]["total_ambiguous_observations"]}</div>
          <div class="metric-label">Ambiguous Crossings</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Sidebar: Player Roster & Analytics -->
  <div>
    <div class="card">
      <div class="card-title">
        <span>Player Roster Analytics</span>
        <span>{len(analytics["player_roster"])} Detected</span>
      </div>
      <div class="roster-list" id="rosterList"></div>
    </div>

    <!-- Player Association & Target Selection (TL-004) -->
    <div class="card">
      <div class="card-title">Player Association & Target Selection (TL-004)</div>
      <div class="review-box">
        <input type="text" id="assocPlayerId" placeholder="Confirmed Player ID (e.g. P07, Player_Smith)" class="review-select">
        <input type="text" id="assocPlayerName" placeholder="Player Name (e.g. Alex Morgan)" class="review-select">
        <label style="display:flex;align-items:center;gap:6px;font-size:12px;margin-bottom:8px;cursor:pointer;">
          <input type="checkbox" id="assocIsTarget"> Mark as Target Participant
        </label>
        <button id="saveAssocBtn" style="width:100%;">Record Player Association</button>
      </div>
    </div>

    <!-- Level 1 Coach Attempt Review (TL-005.2) -->
    <div class="card">
      <div class="card-title">Level 1 Coach Annotation (TL-005.2)</div>
      <div class="review-box">
        <select id="reviewOutcome" class="review-select">
          <option value="pending">Review Status: Pending</option>
          <option value="conclusive_success">Conclusive: Success</option>
          <option value="conclusive_error">Conclusive: Error / Breakdown</option>
          <option value="inconclusive">Inconclusive (Obscured / Uncertain)</option>
        </select>
        <textarea id="reviewNotes" placeholder="Add coach feedback, objective note, or identity correction..."></textarea>
        <div style="display:flex;gap:8px;margin-top:8px;">
          <button id="addAttemptBtn" style="flex:1;">+ Mark Attempt at Current Frame</button>
          <button id="saveReviewBtn" style="flex:1;">Export Review JSON</button>
        </div>
        <div id="attemptDenominatorSummary" style="margin-top:10px;padding:8px;background:rgba(0,0,0,0.25);border-radius:4px;font-size:12px;color:var(--text-secondary);">
          Conclusive Rate: <strong>Unavailable (0/0)</strong> &bull; Excluded: 0 Inconclusive, 0 Pending
        </div>
        <div id="attemptsList" style="margin-top:8px;display:flex;flex-direction:column;gap:6px;max-height:160px;overflow-y:auto;"></div>
      </div>
    </div>
  </div>
</div>

<details>
  <summary>Inspection Manifest & Analytics Schema (JSON)</summary>
  <pre>{html.escape(json.dumps(manifest, indent=2))}</pre>
</details>

<script>
  const analyticsData = {safe_analytics_json};
  const observations = {safe_observations_json};
  let transcriptData = {safe_transcript_json};

  const canvas = document.getElementById('pitchCanvas');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('frameSlider');
  const timeDisplay = document.getElementById('timeDisplay');
  const playBtn = document.getElementById('playBtn');
  const stepBackBtn = document.getElementById('stepBackBtn');
  const stepFwdBtn = document.getElementById('stepFwdBtn');
  const toggleBoxes = document.getElementById('toggleBoxes');
  const toggleTrails = document.getElementById('toggleTrails');
  const toggleContacts = document.getElementById('toggleContacts');
  const toggleHeatmap = document.getElementById('toggleHeatmap');
  const toggleLoop = document.getElementById('toggleLoop');
  const rosterList = document.getElementById('rosterList');
  const activePlayerLabel = document.getElementById('activePlayerLabel');

  let selectedTrackId = null;
  let isPlaying = false;
  let playInterval = null;

  // Group observations by timestamp
  const framesMap = new Map();
  observations.forEach(obs => {{
    const key = obs.frame.timestamp_s;
    if (!framesMap.has(key)) {{
      framesMap.set(key, {{
        timestamp_s: obs.frame.timestamp_s,
        index: obs.frame.index,
        pts: obs.frame.pts,
        observations: []
      }});
    }}
    framesMap.get(key).observations.push(obs);
  }});

  const frameTimes = Array.from(framesMap.keys()).sort((a, b) => a - b);
  slider.max = Math.max(0, frameTimes.length - 1);

  function getTrackColor(trackId) {{
    const colors = ['#00d084', '#0693e3', '#9b51e0', '#fcb900', '#ff6900', '#eb144c', '#00bfa5', '#795548'];
    return colors[(trackId - 1) % colors.length];
  }}

  // Build Roster Cards
  analyticsData.player_roster.forEach(player => {{
    const card = document.createElement('div');
    card.className = 'player-card' + (player.is_target ? ' is-target' : '');
    card.dataset.trackId = player.local_track_id;

    const color = getTrackColor(player.local_track_id);
    const m = player.movement;
    const distanceText = m.calibrated && m.distance_m !== null
      ? `${{m.distance_m}} m`
      : `${{m.image_pixel_distance_px}} px (uncal.)`;
    const speedText = m.calibrated && m.average_speed_mps !== null
      ? `${{m.average_speed_mps}} m/s`
      : `${{m.average_pixel_speed_px_per_s}} px/s`;

    const maxSpeedText = m.calibrated && m.maximum_windowed_speed_mps !== null
      ? ` (Max: ${{m.maximum_windowed_speed_mps}} m/s)`
      : '';
    const iz = player.intensity_zones;
    const ac = player.activity_classification;
    let zoneHtml = '';
    if (iz && iz.status === 'estimated' && iz.zones) {{
      const hi = iz.high_speed_running_time_s ?? 0;
      const mod = iz.zones.zone_2_moderate?.time_s ?? 0;
      const lo = iz.zones.zone_1_low?.time_s ?? 0;
      const spr = ac?.class_durations_s?.sprinting ?? 0;
      const run = ac?.class_durations_s?.running ?? 0;
      const jog = ac?.class_durations_s?.jogging ?? 0;
      zoneHtml = `<div style="margin-top:6px;font-size:11px;color:var(--text-secondary);background:rgba(0,208,132,0.08);padding:5px 7px;border-radius:4px;line-height:1.4;">⚡ <strong>Zones:</strong> High ${{hi}}s &bull; Mod ${{mod}}s &bull; Low ${{lo}}s<br>🏃 <strong>Activity:</strong> Sprint ${{spr}}s &bull; Run ${{run}}s &bull; Jog ${{jog}}s</div>`;
    }}
    const targetBadgeHtml = player.is_target ? '<span class="quality-badge target-badge">🎯 TARGET</span>' : '';
    const prelimBadgeHtml = player.identity_reviewed
      ? '<span class="quality-badge quality-high">ID REVIEWED</span>'
      : '<span class="quality-badge quality-moderate" title="Preliminary track — player identity not yet confirmed">PRELIMINARY</span>';
    const playerConfirmedHtml = player.confirmed_player_id
      ? `<div style="font-size:12px;color:var(--accent);margin-top:2px;">Player: ${{player.confirmed_player_id}} ${{player.player_name ? `(${{player.player_name}})` : ''}}</div>`
      : `<div style="font-size:11px;color:var(--accent-warn);margin-top:2px;">⚠️ Unconfirmed Local Track (Preliminary Estimate)</div>`;

    card.innerHTML = `
      <div class="player-card-header">
        <div class="track-badge">
          <span class="color-dot" style="background:${{color}}"></span>
          Track #${{player.local_track_id}}
        </div>
        <div>
          ${{targetBadgeHtml}}
          ${{prelimBadgeHtml}}
          <span class="quality-badge quality-${{player.tracking_quality}}">${{player.tracking_quality}}</span>
        </div>
      </div>
      ${{playerConfirmedHtml}}
      <div class="metrics-grid" style="margin-top:6px;">
        <div class="metric-item">
          <div class="metric-val">${{player.visible_duration_s}}s</div>
          <div class="metric-label">Visible Time (${{player.observation_count}} frames)</div>
        </div>
        <div class="metric-item">
          <div class="metric-val">${{distanceText}}</div>
          <div class="metric-label">Total Distance</div>
        </div>
        <div class="metric-item">
          <div class="metric-val">${{speedText}}${{maxSpeedText}}</div>
          <div class="metric-label">Avg / Peak Speed</div>
        </div>
        <div class="metric-item">
          <div class="metric-val">${{player.ambiguous_observations}}</div>
          <div class="metric-label">Ambiguities</div>
        </div>
      </div>
      ${{zoneHtml}}
    `;

    card.addEventListener('click', () => {{
      if (selectedTrackId === player.local_track_id) {{
        selectedTrackId = null;
        card.classList.remove('selected');
        activePlayerLabel.textContent = 'All Players';
        document.getElementById('assocPlayerId').value = '';
        document.getElementById('assocPlayerName').value = '';
        document.getElementById('assocIsTarget').checked = false;
      }} else {{
        document.querySelectorAll('.player-card').forEach(c => c.classList.remove('selected'));
        selectedTrackId = player.local_track_id;
        card.classList.add('selected');
        activePlayerLabel.textContent = `Focused on Track #${{player.local_track_id}}` + (player.is_target ? ' [TARGET]' : '');
        document.getElementById('assocPlayerId').value = player.confirmed_player_id || '';
        document.getElementById('assocPlayerName').value = player.player_name || '';
        document.getElementById('assocIsTarget').checked = !!player.is_target;
      }}
      render();
    }});

    rosterList.appendChild(card);
  }});

  function drawPitchLines() {{
    const w = canvas.width;
    const h = canvas.height;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 2;

    // Pitch perimeter
    ctx.strokeRect(w * 0.05, h * 0.05, w * 0.9, h * 0.9);
    // Halfway line
    ctx.beginPath();
    ctx.moveTo(w * 0.5, h * 0.05);
    ctx.lineTo(w * 0.5, h * 0.95);
    ctx.stroke();
    // Center circle
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.5, h * 0.2, 0, Math.PI * 2);
    ctx.stroke();
  }}

  function renderHeatmap() {{
    const heatmap = (selectedTrackId !== null)
      ? analyticsData.player_roster.find(p => p.local_track_id === selectedTrackId)?.spatial_heatmap
      : analyticsData.team_spatial_heatmap;
    if (!heatmap) return;

    const cellW = canvas.width / heatmap.grid_cols;
    const cellH = canvas.height / heatmap.grid_rows;

    heatmap.density_matrix.forEach((row, r) => {{
      row.forEach((density, c) => {{
        if (density > 0) {{
          const alpha = Math.min(0.7, density * 0.85);
          ctx.fillStyle = `rgba(255, 77, 79, ${{alpha}})`;
          ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }}
      }});
    }});
  }}

  const sourceVideo = document.getElementById('sourceVideo');
  const toggleVideo = document.getElementById('toggleVideo');
  const visibilityStrip = document.getElementById('visibilityStrip');
  let videoAvailable = false;
  if (sourceVideo) {{
    sourceVideo.addEventListener('loadeddata', () => {{
      videoAvailable = true;
      sourceVideo.style.display = toggleVideo.checked ? 'block' : 'none';
      render();
    }});
    sourceVideo.addEventListener('error', () => {{
      videoAvailable = false;
      sourceVideo.style.display = 'none';
    }});
  }}
  if (toggleVideo) {{
    toggleVideo.addEventListener('change', () => {{
      if (sourceVideo && videoAvailable) {{
        sourceVideo.style.display = toggleVideo.checked ? 'block' : 'none';
      }}
      render();
    }});
  }}

  function renderVisibilityStrip() {{
    if (!visibilityStrip || frameTimes.length === 0) return;
    visibilityStrip.innerHTML = '';
    const minT = frameTimes[0];
    const maxT = frameTimes[frameTimes.length - 1];
    const span = Math.max(0.001, maxT - minT);
    const activeObsTimes = (selectedTrackId !== null)
      ? (analyticsData.player_roster.find(p => p.local_track_id === selectedTrackId)?.trajectory.map(pt => pt.timestamp_s) || [])
      : frameTimes;
    activeObsTimes.forEach(t => {{
      const pct = ((t - minT) / span) * 100;
      const tick = document.createElement('div');
      tick.style.cssText = `position:absolute;left:${{Math.min(99, Math.max(0, pct))}}%;top:0;width:${{Math.max(1.2, 100 / frameTimes.length)}}%;height:100%;background:#00d084;`;
      visibilityStrip.appendChild(tick);
    }});
  }}

  function render() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const frameIdx = parseInt(slider.value, 10);
    const currTime = frameTimes[frameIdx] ?? 0;
    if (videoAvailable && toggleVideo && toggleVideo.checked) {{
      if (Math.abs(sourceVideo.currentTime - currTime) > 0.015) {{
        sourceVideo.currentTime = currTime;
      }}
    }} else {{
      drawPitchLines();
    }}
    renderVisibilityStrip();

    if (toggleHeatmap.checked) {{
      renderHeatmap();
    }}

    const currentFrame = framesMap.get(currTime);

    timeDisplay.textContent = currentFrame
      ? `Frame ${{currentFrame.index}} (${{currTime.toFixed(2)}}s) [PTS ${{currentFrame.pts ?? 'N/A'}}]`
      : `0.00s`;

    // Draw Motion Trails
    if (toggleTrails.checked) {{
      analyticsData.player_roster.forEach(player => {{
        if (selectedTrackId !== null && player.local_track_id !== selectedTrackId) return;
        const color = player.is_target ? '#ff6b35' : getTrackColor(player.local_track_id);
        const subTrajectory = player.trajectory.filter(pt => pt.timestamp_s <= currTime);
        if (subTrajectory.length > 1) {{
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = (player.local_track_id === selectedTrackId || player.is_target) ? 3 : 2;
          ctx.globalAlpha = 0.6;
          subTrajectory.forEach((pt, i) => {{
            const [gx, gy] = pt.ground_contact_px;
            if (i === 0) ctx.moveTo(gx, gy);
            else ctx.lineTo(gx, gy);
          }});
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }}
      }});
    }}

    // Draw Current Frame Observations
    if (currentFrame && toggleBoxes.checked) {{
      currentFrame.observations.forEach(obs => {{
        if (selectedTrackId !== null && obs.local_track_id !== selectedTrackId) return;
        const [x1, y1, x2, y2] = obs.bbox_xyxy;
        const playerObj = analyticsData.player_roster.find(p => p.local_track_id === obs.local_track_id);
        const isTarget = playerObj ? playerObj.is_target : false;
        const color = isTarget ? '#ff6b35' : getTrackColor(obs.local_track_id);

        ctx.strokeStyle = color;
        ctx.lineWidth = (obs.local_track_id === selectedTrackId || isTarget) ? 3 : 2;
        if (obs.polygon && obs.polygon.length > 2) {{
          ctx.beginPath();
          ctx.moveTo(obs.polygon[0][0], obs.polygon[0][1]);
          for (let pi = 1; pi < obs.polygon.length; pi++) {{
            ctx.lineTo(obs.polygon[pi][0], obs.polygon[pi][1]);
          }}
          ctx.closePath();
          ctx.fillStyle = isTarget ? 'rgba(255, 107, 53, 0.25)' : 'rgba(0, 208, 132, 0.2)';
          ctx.fill();
          ctx.stroke();
        }}
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Track ID Tag
        ctx.fillStyle = color;
        const tagWidth = isTarget ? 90 : 70;
        ctx.fillRect(x1, Math.max(0, y1 - 18), tagWidth, 18);
        ctx.fillStyle = '#0b1319';
        ctx.font = 'bold 11px sans-serif';
        const label = (isTarget ? '🎯 ' : '') + `T#${{obs.local_track_id}} ${{obs.association_state === 'ambiguous' ? '⚠️' : ''}}`;
        ctx.fillText(label, x1 + 4, Math.max(12, y1 - 4));

        // Ground Contact Point
        if (toggleContacts.checked) {{
          const gx = (x1 + x2) / 2;
          const gy = y2;
          ctx.beginPath();
          ctx.arc(gx, gy, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#ff4d4f';
          ctx.fill();
        }}
      }});
    }}
  }}

  // Playback Controls
  function play() {{
    isPlaying = true;
    playBtn.textContent = '⏸ Pause';
    playInterval = setInterval(() => {{
      let next = parseInt(slider.value, 10) + 1;
      if (next >= frameTimes.length) {{
        if (toggleLoop.checked) {{
          next = 0;
        }} else {{
          pause();
          return;
        }}
      }}
      slider.value = next;
      render();
    }}, 200);
  }}

  function pause() {{
    isPlaying = false;
    playBtn.textContent = '▶ Play';
    if (playInterval) clearInterval(playInterval);
  }}

  playBtn.addEventListener('click', () => isPlaying ? pause() : play());
  stepBackBtn.addEventListener('click', () => {{
    pause();
    slider.value = Math.max(0, parseInt(slider.value, 10) - 1);
    render();
  }});
  stepFwdBtn.addEventListener('click', () => {{
    pause();
    slider.value = Math.min(frameTimes.length - 1, parseInt(slider.value, 10) + 1);
    render();
  }});
  slider.addEventListener('input', () => {{
    pause();
    render();
  }});

  [toggleBoxes, toggleTrails, toggleContacts, toggleHeatmap].forEach(el => {{
    el.addEventListener('change', render);
  }});

  // Save Association (TL-004)
  document.getElementById('saveAssocBtn').addEventListener('click', () => {{
    if (!selectedTrackId) {{
      alert('Please click and select a player track card first!');
      return;
    }}
    const pId = document.getElementById('assocPlayerId').value.trim();
    if (!pId) {{
      alert('Please enter a confirmed Player ID.');
      return;
    }}
    const pName = document.getElementById('assocPlayerName').value.trim();
    const isTarget = document.getElementById('assocIsTarget').checked;

    const assocPayload = {{
      schema_version: "touchline.association-ledger.v1",
      run_id: analyticsData.run_id,
      local_track_id: selectedTrackId,
      player_id: pId,
      player_name: pName || null,
      is_target: isTarget,
      timestamp_utc: new Date().toISOString()
    }};
    const blob = new Blob([JSON.stringify(assocPayload, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `association-${{analyticsData.run_id?.slice(0, 8) ?? 'run'}}-track-${{selectedTrackId}}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }});

  // Attempt list and denominator tracking (TL-005.2)
  const recordedAttempts = [];
  function updateAttemptsUI() {{
    const summaryEl = document.getElementById('attemptDenominatorSummary');
    const listEl = document.getElementById('attemptsList');
    let succ = 0, err = 0, inc = 0, pend = 0;
    recordedAttempts.forEach(a => {{
      if (a.outcome === 'conclusive_success') succ++;
      else if (a.outcome === 'conclusive_error') err++;
      else if (a.outcome === 'inconclusive') inc++;
      else pend++;
    }});
    const denom = succ + err;
    const rateStr = denom > 0 ? `${{succ}}/${{denom}} (${{((succ / denom) * 100).toFixed(1)}}%)` : `Unavailable (0/0)`;
    summaryEl.innerHTML = `Conclusive Rate: <strong style="color:var(--accent);">${{rateStr}}</strong> &bull; Excluded: ${{inc}} Inconclusive, ${{pend}} Pending`;
    listEl.innerHTML = '';
    recordedAttempts.forEach((a, idx) => {{
      const row = document.createElement('div');
      row.style.cssText = 'padding:6px 8px;background:var(--bg-card-sub);border:1px solid var(--border);border-radius:4px;font-size:12px;cursor:pointer;display:flex;justify-content:space-between;';
      row.innerHTML = `<span>#${{idx + 1}} [T#${{a.track_id ?? 'All'}} @ ${{a.timestamp_s.toFixed(2)}}s / PTS ${{a.pts ?? 'N/A'}}]</span><strong>${{a.outcome}}</strong>`;
      row.addEventListener('click', () => {{
        pause();
        slider.value = a.frame_slider_idx;
        render();
      }});
      listEl.appendChild(row);
    }});
  }}

  const addAttemptBtn = document.getElementById('addAttemptBtn');
  if (addAttemptBtn) {{
    addAttemptBtn.addEventListener('click', () => {{
      const frameIdx = parseInt(slider.value, 10);
      const currTime = frameTimes[frameIdx] ?? 0;
      const currentFrame = framesMap.get(currTime);
      recordedAttempts.push({{
        attempt_id: recordedAttempts.length + 1,
        track_id: selectedTrackId,
        frame_slider_idx: frameIdx,
        timestamp_s: currTime,
        pts: currentFrame ? currentFrame.pts : null,
        outcome: document.getElementById('reviewOutcome').value,
        notes: document.getElementById('reviewNotes').value.trim(),
        timestamp_utc: new Date().toISOString()
      }});
      updateAttemptsUI();
    }});
  }}

  // Save Review / Export JSON (TL-005.2)
  document.getElementById('saveReviewBtn').addEventListener('click', () => {{
    const outcome = document.getElementById('reviewOutcome').value;
    const notes = document.getElementById('reviewNotes').value;
    const reviewData = {{
      schema_version: "touchline.review-annotation.v1",
      run_id: analyticsData.run_id,
      track_id: selectedTrackId,
      outcome: outcome,
      reviewer_notes: notes,
      timestamp_utc: new Date().toISOString()
    }};
    const blob = new Blob([JSON.stringify(reviewData, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `review-${{analyticsData.run_id?.slice(0, 8) ?? 'run'}}-track-${{selectedTrackId ?? 'all'}}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }});

  // Coach Bluetooth Collar Mic & Sarvam AI Transcript Workflow
  const micDeviceSelect = document.getElementById('micDeviceSelect');
  const sarvamLangSelect = document.getElementById('sarvamLangSelect');
  const refreshMicBtn = document.getElementById('refreshMicBtn');
  const startMicBtn = document.getElementById('startMicBtn');
  const stopMicBtn = document.getElementById('stopMicBtn');
  const markLiveCueBtn = document.getElementById('markLiveCueBtn');
  const liveCueNoteInput = document.getElementById('liveCueNoteInput');
  const uploadAudioFileInput = document.getElementById('uploadAudioFileInput');
  const micStatusBanner = document.getElementById('micStatusBanner');
  const transcriptTimelineStrip = document.getElementById('transcriptTimelineStrip');
  const transcriptSegmentsList = document.getElementById('transcriptSegmentsList');

  let mediaRecorder = null;
  let audioChunks = [];
  let recVideoAnchorStartS = 0.0;
  let recMonotonicStartMs = 0.0;
  let liveCuesDuringRecording = [];

  function getRunDirIdFromLocation() {{
    const parts = window.location.pathname.split('/').filter(Boolean);
    const runsIdx = parts.indexOf('runs');
    if (runsIdx >= 0 && parts.length > runsIdx + 1) {{
      return parts[runsIdx + 1];
    }}
    return analyticsData.run_id || 'calibrated-snmot097';
  }}

  function seekToVideoTime(targetTimeS) {{
    if (frameTimes.length === 0) return;
    pause();
    let bestIdx = 0;
    let bestDiff = Math.abs(frameTimes[0] - targetTimeS);
    for (let i = 1; i < frameTimes.length; i++) {{
      const d = Math.abs(frameTimes[i] - targetTimeS);
      if (d < bestDiff) {{
        bestDiff = d;
        bestIdx = i;
      }}
    }}
    slider.value = bestIdx;
    if (videoAvailable && sourceVideo) {{
      sourceVideo.currentTime = targetTimeS;
    }}
    render();
  }}

  function renderTranscriptTimeline() {{
    if (!transcriptTimelineStrip || !transcriptSegmentsList) return;
    transcriptTimelineStrip.innerHTML = '';
    transcriptSegmentsList.innerHTML = '';

    const segments = (transcriptData && Array.isArray(transcriptData.segments)) ? transcriptData.segments : [];
    const cues = (transcriptData && Array.isArray(transcriptData.manual_cues)) ? transcriptData.manual_cues : [];

    const minT = frameTimes.length > 0 ? frameTimes[0] : 0.0;
    const maxT = frameTimes.length > 0 ? Math.max(frameTimes[frameTimes.length - 1], minT + 1.0) : 10.0;
    const span = Math.max(0.1, maxT - minT);

    segments.forEach((seg, idx) => {{
      const vStart = Number(seg.video_start_s ?? 0);
      const vEnd = Number(seg.video_end_s ?? (vStart + 1.0));
      const leftPct = Math.min(98, Math.max(0, ((vStart - minT) / span) * 100));
      const widthPct = Math.min(100 - leftPct, Math.max(2.0, ((vEnd - vStart) / span) * 100));

      const marker = document.createElement('div');
      marker.title = `[${{vStart.toFixed(2)}}s] ${{seg.text}}`;
      marker.style.cssText = `position:absolute;left:${{leftPct}}%;top:1px;width:${{widthPct}}%;height:10px;background:#ff6b35;border-radius:2px;cursor:pointer;opacity:0.9;`;
      marker.addEventListener('click', () => seekToVideoTime(vStart));
      transcriptTimelineStrip.appendChild(marker);

      const row = document.createElement('div');
      row.style.cssText = 'padding:8px 10px;background:var(--bg-card-sub);border-left:3px solid #ff6b35;border-radius:4px;font-size:12px;cursor:pointer;display:flex;flex-direction:column;gap:3px;';
      row.innerHTML = `
        <div style="display:flex;justify-content:space-between;color:var(--text-secondary);font-family:monospace;font-size:11px;">
          <span>🎬 Video: <strong style="color:var(--accent);">${{vStart.toFixed(2)}}s–${{vEnd.toFixed(2)}}s</strong> (PTS: ${{seg.nearest_frame_pts ?? 'N/A'}} | Frame #${{seg.nearest_frame_index ?? 'N/A'}})</span>
          <span>🎙️ Audio: +${{Number(seg.audio_start_s ?? 0).toFixed(2)}}s &bull; ${{seg.speaker_label || 'Coach'}}</span>
        </div>
        <div style="color:var(--text-primary);font-size:13px;font-weight:500;">"${{seg.text}}"</div>
      `;
      row.addEventListener('click', () => seekToVideoTime(vStart));
      transcriptSegmentsList.appendChild(row);
    }});

    cues.forEach((cue, idx) => {{
      const vStart = Number(cue.video_start_s ?? 0);
      const leftPct = Math.min(98, Math.max(0, ((vStart - minT) / span) * 100));
      const marker = document.createElement('div');
      marker.title = `Cue @ ${{vStart.toFixed(2)}}s: ${{cue.note}}`;
      marker.style.cssText = `position:absolute;left:${{leftPct}}%;top:0;width:4px;height:14px;background:#0693e3;cursor:pointer;`;
      marker.addEventListener('click', () => seekToVideoTime(vStart));
      transcriptTimelineStrip.appendChild(marker);

      const row = document.createElement('div');
      row.style.cssText = 'padding:6px 10px;background:rgba(6,147,227,0.12);border-left:3px solid #0693e3;border-radius:4px;font-size:12px;cursor:pointer;display:flex;justify-content:space-between;';
      row.innerHTML = `<span>📌 <strong>Coach Cue:</strong> ${{cue.note}} ${{cue.target_track_id ? `(T#${{cue.target_track_id}})` : ''}}</span><span style="font-family:monospace;">Video ${{vStart.toFixed(2)}}s (PTS ${{cue.nearest_frame_pts ?? 'N/A'}})</span>`;
      row.addEventListener('click', () => seekToVideoTime(vStart));
      transcriptSegmentsList.appendChild(row);
    }});

    if (segments.length === 0 && cues.length === 0) {{
      transcriptSegmentsList.innerHTML = `<div style="font-size:12px;color:var(--text-secondary);padding:6px;">No coach transcript recorded for this run yet. Use <strong>🔴 Record Collar Mic</strong> or <strong>📂 Upload Audio</strong> above to transcribe with Sarvam AI.</div>`;
    }}
  }}

  async function refreshAudioDevices() {{
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {{
      micStatusBanner.textContent = '⚠️ Browser audio enumeration unavailable (requires localhost or HTTPS). You can still upload an audio file.';
      return;
    }}
    try {{
      // Request brief permission so labels (like Bluetooth Collar Mic) are visible
      const tempStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      tempStream.getTracks().forEach(t => t.stop());
    }} catch (_) {{}}
    try {{
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(d => d.kind === 'audioinput');
      micDeviceSelect.innerHTML = '';
      if (audioInputs.length === 0) {{
        micDeviceSelect.innerHTML = '<option value="">Default Microphone</option>';
      }} else {{
        audioInputs.forEach((dev, i) => {{
          const opt = document.createElement('option');
          opt.value = dev.deviceId;
          const label = dev.label || `Microphone #${{i + 1}}`;
          opt.textContent = `🎙️ ${{label}}`;
          if (/bluetooth|collar|wireless|headset|buds|airpods|hands-free/i.test(label)) {{
            opt.selected = true;
          }}
          micDeviceSelect.appendChild(opt);
        }});
      }}
      micStatusBanner.innerHTML = `✅ Found <strong>${{audioInputs.length}}</strong> audio input device(s). Select your Bluetooth Collar Mic and click <strong>🔴 Record Collar Mic</strong>.`;
    }} catch (err) {{
      micStatusBanner.textContent = '⚠️ Could not list audio devices: ' + err;
    }}
  }}

  if (refreshMicBtn) {{
    refreshMicBtn.addEventListener('click', refreshAudioDevices);
  }}

  async function uploadAndTranscribeAudioBlob(blob, anchorVideoS, monotonicMs, deviceLabel, cuesList) {{
    const runDirId = encodeURIComponent(getRunDirIdFromLocation());
    const lang = encodeURIComponent(sarvamLangSelect ? sarvamLangSelect.value : 'en-IN');
    const devLabel = encodeURIComponent(deviceLabel || 'Bluetooth Collar Mic');
    const cuesJson = encodeURIComponent(JSON.stringify(cuesList || []));

    micStatusBanner.innerHTML = `⏳ Sending coach audio (${{(blob.size / 1024).toFixed(1)}} KB) to <strong>Sarvam AI Speech-to-Text</strong> and aligning to video anchor <code>${{anchorVideoS.toFixed(2)}}s</code>...`;
    try {{
      const arrayBuf = await blob.arrayBuffer();
      const url = `/api/audio/transcribe?run_id=${{runDirId}}&video_anchor_start_s=${{anchorVideoS}}&monotonic_start_ms=${{monotonicMs}}&language_code=${{lang}}&device_label=${{devLabel}}&cues=${{cuesJson}}`;
      const resp = await fetch(url, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/octet-stream' }},
        body: arrayBuf
      }});
      const data = await resp.json();
      if (!resp.ok || data.error) {{
        micStatusBanner.innerHTML = `❌ <strong>Sarvam AI Error:</strong> ${{data.error || resp.statusText}} (Check that your Sarvam API Key is saved at the top of Touchline Studio!)`;
      }} else {{
        transcriptData = data.transcript;
        renderTranscriptTimeline();
        micStatusBanner.innerHTML = `✅ <strong>Sarvam AI Transcription Complete!</strong> Aligned <strong>${{(transcriptData.segments || []).length}}</strong> segment(s) &amp; <strong>${{(transcriptData.manual_cues || []).length}}</strong> cue(s) to video timeline. Click any quote below to jump to that video frame!`;
      }}
    }} catch (err) {{
      micStatusBanner.innerHTML = `❌ Upload/Transcription failed: ${{err}}`;
    }}
  }}

  if (startMicBtn && stopMicBtn) {{
    startMicBtn.addEventListener('click', async () => {{
      try {{
        const selectedDeviceId = micDeviceSelect ? micDeviceSelect.value : '';
        const constraints = {{
          audio: selectedDeviceId ? {{ deviceId: {{ exact: selectedDeviceId }}, echoCancellation: true, noiseSuppression: true }} : true
        }};
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        audioChunks = [];
        liveCuesDuringRecording = [];
        const frameIdx = parseInt(slider.value, 10);
        recVideoAnchorStartS = (videoAvailable && sourceVideo && !sourceVideo.paused)
          ? sourceVideo.currentTime
          : (frameTimes[frameIdx] ?? 0.0);
        recMonotonicStartMs = performance.now();

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => {{
          if (e.data && e.data.size > 0) audioChunks.push(e.data);
        }};
        mediaRecorder.onstop = async () => {{
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(audioChunks, {{ type: mediaRecorder.mimeType || 'audio/webm' }});
          const devName = micDeviceSelect && micDeviceSelect.options[micDeviceSelect.selectedIndex]
            ? micDeviceSelect.options[micDeviceSelect.selectedIndex].textContent
            : 'Bluetooth Collar Mic';
          await uploadAndTranscribeAudioBlob(blob, recVideoAnchorStartS, recMonotonicStartMs, devName, liveCuesDuringRecording);
        }};

        mediaRecorder.start(250);
        startMicBtn.disabled = true;
        startMicBtn.style.opacity = '0.5';
        stopMicBtn.disabled = false;
        stopMicBtn.style.opacity = '1.0';
        if (!isPlaying) play();
        micStatusBanner.innerHTML = `🔴 <strong>RECORDING COLLAR MIC...</strong> Video Anchor: <code>${{recVideoAnchorStartS.toFixed(2)}}s</code> (Monotonic: <code>${{Math.round(recMonotonicStartMs)}}ms</code>). Speak coaching instructions now, then click <strong>⏹️ Stop &amp; Transcribe (Sarvam AI)</strong>.`;
      }} catch (err) {{
        micStatusBanner.innerHTML = `❌ Microphone access error: ${{err}}. Click 🔄 Devices or use 📂 Upload Audio.`;
      }}
    }});

    stopMicBtn.addEventListener('click', () => {{
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
        pause();
        mediaRecorder.stop();
        startMicBtn.disabled = false;
        startMicBtn.style.opacity = '1.0';
        stopMicBtn.disabled = true;
        stopMicBtn.style.opacity = '0.5';
      }}
    }});
  }}

  const pttMicBtn = document.getElementById('pttMicBtn');
  let isPttRecording = false;

  async function startPttRecording() {{
    if (isPttRecording) return;
    try {{
      const selectedDeviceId = micDeviceSelect ? micDeviceSelect.value : '';
      const constraints = {{
        audio: selectedDeviceId ? {{ deviceId: {{ exact: selectedDeviceId }}, echoCancellation: true, noiseSuppression: true }} : true
      }};
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      audioChunks = [];
      liveCuesDuringRecording = [];
      const frameIdx = parseInt(slider.value, 10);
      recVideoAnchorStartS = (videoAvailable && sourceVideo && !sourceVideo.paused)
        ? sourceVideo.currentTime
        : (frameTimes[frameIdx] ?? 0.0);
      recMonotonicStartMs = performance.now();

      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {{
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      }};
      mediaRecorder.onstop = async () => {{
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(audioChunks, {{ type: mediaRecorder.mimeType || 'audio/webm' }});
        const devName = micDeviceSelect && micDeviceSelect.options[micDeviceSelect.selectedIndex]
          ? micDeviceSelect.options[micDeviceSelect.selectedIndex].textContent
          : 'Bluetooth Collar Mic (PTT)';
        await uploadAndTranscribeAudioBlob(blob, recVideoAnchorStartS, recMonotonicStartMs, devName, liveCuesDuringRecording);
      }};

      mediaRecorder.start(100);
      isPttRecording = true;
      pttMicBtn.style.background = '#e65100';
      pttMicBtn.style.borderColor = '#ff9800';
      pttMicBtn.textContent = '🎙️ Speaking... Release to Send';
      micStatusBanner.innerHTML = `🎙️ <strong>PUSH-TO-TALK ACTIVE:</strong> Recording snippet at Video Time <code>${{recVideoAnchorStartS.toFixed(2)}}s</code>. Release button to transcribe and pin to video.`;
    }} catch (err) {{
      micStatusBanner.innerHTML = `❌ PTT error: ${{err}}`;
    }}
  }}

  function stopPttRecording() {{
    if (!isPttRecording) return;
    isPttRecording = false;
    pttMicBtn.style.background = '#1b384f';
    pttMicBtn.style.borderColor = '#1e90ff';
    pttMicBtn.textContent = '🎙️ Hold to Speak (Push-to-Talk)';
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
      mediaRecorder.stop();
    }}
  }}

  if (pttMicBtn) {{
    pttMicBtn.addEventListener('mousedown', startPttRecording);
    pttMicBtn.addEventListener('mouseup', stopPttRecording);
    pttMicBtn.addEventListener('mouseleave', stopPttRecording);
    pttMicBtn.addEventListener('touchstart', (e) => {{ e.preventDefault(); startPttRecording(); }});
    pttMicBtn.addEventListener('touchend', (e) => {{ e.preventDefault(); stopPttRecording(); }});
  }}

  // Bluetooth Headset / Earphone Button Controller (MediaSession API)
  // When coach clicks their Bluetooth earphone button (Play/Pause/Track), it toggles PTT recording hands-free!
  function togglePttRecording() {{
    if (isPttRecording) {{
      stopPttRecording();
      if (navigator.vibrate) navigator.vibrate([80]);
    }} else {{
      startPttRecording();
      if (navigator.vibrate) navigator.vibrate([150]);
    }}
  }}

  if ('mediaSession' in navigator) {{
    try {{
      navigator.mediaSession.metadata = new MediaMetadata({{
        title: 'Touchline Coach Mic',
        artist: 'Meta Muse Voice Transcribe',
        album: 'Match Voiceover'
      }});
      const mediaActions = ['play', 'pause', 'stop', 'nexttrack', 'previoustrack'];
      mediaActions.forEach(action => {{
        try {{
          navigator.mediaSession.setActionHandler(action, () => {{
            togglePttRecording();
          }});
        }} catch (e) {{}}
      }});
    }} catch (e) {{
      console.warn('MediaSession handler registration:', e);
    }}
  }}

  // Also support Bluetooth keyboard/clicker remote triggers (Space, MediaPlayPause, or VolumeUp)
  window.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space' || e.key === 'MediaPlayPause' || e.key === 'AudioVolumeUp') {{
      if (!isPttRecording) {{
        startPttRecording();
        if (navigator.vibrate) navigator.vibrate([120]);
      }}
    }}
  }});
  window.addEventListener('keyup', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space' || e.key === 'MediaPlayPause' || e.key === 'AudioVolumeUp') {{
      if (isPttRecording) {{
        stopPttRecording();
        if (navigator.vibrate) navigator.vibrate([60]);
      }}
    }}
  }});

  if (markLiveCueBtn) {{
    markLiveCueBtn.addEventListener('click', () => {{
      const noteText = (liveCueNoteInput && liveCueNoteInput.value.trim()) || 'Coach live timestamp cue';
      const frameIdx = parseInt(slider.value, 10);
      const currVideoT = frameTimes[frameIdx] ?? 0.0;
      const currentFrame = framesMap.get(currVideoT);
      const audioOffsetS = (mediaRecorder && mediaRecorder.state === 'recording')
        ? Math.max(0, (performance.now() - recMonotonicStartMs) / 1000.0)
        : currVideoT;
      const cueObj = {{
        audio_offset_s: Number(audioOffsetS.toFixed(3)),
        video_start_s: Number(currVideoT.toFixed(3)),
        nearest_frame_pts: currentFrame ? currentFrame.pts : null,
        nearest_frame_index: currentFrame ? currentFrame.index : null,
        note: noteText,
        target_track_id: selectedTrackId
      }};
      liveCuesDuringRecording.push(cueObj);
      if (!transcriptData.manual_cues) transcriptData.manual_cues = [];
      transcriptData.manual_cues.push(cueObj);
      if (liveCueNoteInput) liveCueNoteInput.value = '';
      renderTranscriptTimeline();
    }});
  }}

  if (uploadAudioFileInput) {{
    uploadAudioFileInput.addEventListener('change', async () => {{
      if (!uploadAudioFileInput.files || uploadAudioFileInput.files.length === 0) return;
      const file = uploadAudioFileInput.files[0];
      const frameIdx = parseInt(slider.value, 10);
      const anchorS = frameTimes[frameIdx] ?? 0.0;
      await uploadAndTranscribeAudioBlob(file, anchorS, performance.now(), `Uploaded File (${{file.name}})`, liveCuesDuringRecording);
    }});
  }}

  // Initial draw
  render();
  renderTranscriptTimeline();
</script>
</body>
</html>
'''
