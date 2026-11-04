'use strict';
// Explicit, stateless development fixture. It is neither a backend nor persistent storage.
(function (root) {
  function response(status, data) { return {ok: status >= 200 && status < 300, status, json: async () => structuredClone(data)}; }
  function createSyntheticFixtureTransport({failures = {}} = {}) {
    const calls = [];
    async function fetchMock(url, options = {}) {
      const path = new URL(url, 'http://127.0.0.1:8000').pathname;
      const method = options.method || 'GET';
      const key = `${method} ${path}`;
      calls.push({method, path, headers: {...options.headers}, body: options.body});
      if (failures[key]) {
        const {status, code} = failures[key];
        return response(status, {error: {code, status, message: 'Synthetic fixture failure'}});
      }
      const role = options.headers?.['X-Touchline-Mock-Role'];
      if (path.includes('/guardian/reports/') && role !== 'guardian') return response(403, {error: {code: 'FORBIDDEN_SCOPE', status: 403}});
      if (method === 'POST' && role !== 'coach') return response(403, {error: {code: 'FORBIDDEN_SCOPE', status: 403}});
      const body = options.body ? JSON.parse(options.body) : {};
      if (method === 'POST' && path.endsWith('/association-reviews')) return response(201, {
        review_id: `synthetic_asv_${body.run_id}_${body.local_track_id}`, school_id: 'synthetic_sch_1', run_id: body.run_id,
        local_track_id: body.local_track_id, canonical_player_id: body.canonical_player_id,
        decision: body.decision, revision: 1, reviewed_by_user_id: 'synthetic_coach_1',
        created_at_utc: '2026-10-31T00:00:00Z',
      });
      if (method === 'POST' && path.endsWith('/corrections')) return response(201, {
        correction_id: 'synthetic_cor_1', status: 'active', revision: 1,
      });
      if (method === 'POST' && path.endsWith('/evidence')) return response(201, {
        evidence_id: 'synthetic_evi_1', correction_id: 'synthetic_cor_1', review_status: 'pending', revision: 1,
      });
      if (method === 'POST' && path.endsWith('/assessments')) return response(201, {
        assessment_id: 'synthetic_asm_1', report_id: 'synthetic_rep_1', revision: 1,
        status: 'pending_review', content_hash: 'sha256_synthetic_fixture',
      });
      if (method === 'POST' && path.endsWith('/approve')) return response(201, {
        approval_id: 'synthetic_appr_1', report_id: path.split('/').at(-2), revision: 1,
        status: 'approved', content_hash: body.expected_content_hash,
        approved_by_user_id: 'synthetic_coach_1', approved_at_utc: '2026-10-31T00:00:00Z',
      });
      if (method === 'GET' && path.includes('/guardian/reports/')) return response(200, {
        report_id: path.split('/').at(-1),
        player: {display_name: 'Synthetic player', jersey: '00', team_name: 'Synthetic team'},
        assessment: {objective: 'Synthetic criterion', status: 'still_needs_work', coach_summary: 'Synthetic sample only', approved_at_utc: '2026-10-31T00:00:00Z'},
        evidence_moments: [],
      });
      return response(404, {error: {code: 'NOT_FOUND', status: 404}});
    }
    return Object.freeze({fetchMock, calls, label: 'SYNTHETIC FIXTURE — not saved to backend'});
  }
  const exported = Object.freeze({createSyntheticFixtureTransport});
  root.TouchlinePlayerDevelopmentMock = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
