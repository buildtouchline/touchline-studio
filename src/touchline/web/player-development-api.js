'use strict';
// Provisional /api/v1 client; endpoint examples have no confirmed OpenAPI contract.
(function (root) {
  const messages = Object.freeze({
    UNAUTHENTICATED: 'Your session has expired. Sign in again.',
    FORBIDDEN_SCOPE: 'You do not have access to this record.',
    CONSENT_REVOKED: 'Family access is suspended.',
    NOT_FOUND: 'This record is unavailable.',
    PRECONDITION_FAILED: 'This record changed. Reload it before saving.',
    ASSOCIATION_MUTATED: 'The player association changed. Review it again.',
    ASSOCIATION_UNCONFIRMED: 'Confirm the player association before continuing.',
    CROSS_SESSION_MISMATCH: 'The two sessions do not have the same reviewed player.',
    MEDIA_OUT_OF_BOUNDS: 'Select a moment within the available video.',
    TIMEOUT: 'The request timed out. Try again.',
    CANCELLED: 'The request was cancelled.',
    NETWORK_ERROR: 'The service could not be reached. Try again.',
    INVALID_RESPONSE: 'The service returned an invalid response.',
  });
  class ApiError extends Error {
    constructor(code, status = 0, target = null) {
      super(messages[code] || 'The request could not be completed.');
      this.name = 'ApiError'; this.code = code; this.status = status; this.target = target;
    }
  }
  function required(value, label) {
    if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${label} is required`);
    return value;
  }
  function id(value, label) { return encodeURIComponent(required(value, label)); }
  function natural(value, label) {
    if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${label} must be a nonnegative integer`);
  }
  function revision(value) { required(value, 'revision precondition'); return value; }
  function object(value, label) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
    return value;
  }
  function result(value, fields) {
    if (!value || typeof value !== 'object' || Array.isArray(value) ||
      fields.some(field => typeof value[field] !== 'string' || !value[field])) throw new ApiError('INVALID_RESPONSE');
    return value;
  }
  function loopback(origin) {
    try { return ['127.0.0.1', 'localhost', '[::1]'].includes(new URL(origin).hostname); }
    catch { return false; }
  }
  function createClient(config = {}) {
    const mode = config.mode || 'production';
    if (!['production', 'local-test'].includes(mode)) throw new TypeError('Unknown API mode');
    const baseUrl = config.baseUrl == null ? '' : required(config.baseUrl, 'baseUrl').replace(/\/$/, '');
    if (mode === 'production' && /^http:\/\//i.test(baseUrl) && !loopback(baseUrl)) throw new TypeError('Production bearer tokens require HTTPS or same-origin transport');
    const timeoutMs = config.timeoutMs == null ? 12000 : config.timeoutMs;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new TypeError('timeoutMs must be positive');
    if (config.mockActor && (mode !== 'local-test' || !loopback(baseUrl))) throw new TypeError('Mock actor headers require explicit local-test mode and a loopback baseUrl');
    if (config.mockActor && config.getToken) throw new TypeError('Choose either a mock actor or a bearer token provider');
    if (config.mockFetch && mode !== 'local-test') throw new TypeError('Mock transport requires explicit local-test mode');
    if (mode === 'production' && typeof config.getToken !== 'function') throw new TypeError('Production mode requires a bearer token provider');
    if (mode === 'local-test' && !config.mockActor && typeof config.getToken !== 'function') throw new TypeError('Local test mode requires an explicit actor or token provider');
    const fetchImpl = config.mockFetch || config.fetchImpl || root.fetch?.bind(root);
    if (typeof fetchImpl !== 'function') throw new TypeError('fetch transport is required');
    const actor = config.mockActor;
    if (actor) {
      required(actor.id, 'mock actor ID'); required(actor.role, 'mock role'); required(actor.schoolId, 'mock school ID');
    }
    async function request(path, {method = 'GET', body, signal, idempotencyKey, ifMatch} = {}) {
      if (signal?.aborted) throw new ApiError('CANCELLED');
      if (idempotencyKey !== undefined) required(idempotencyKey, 'idempotency key');
      if (ifMatch !== undefined) required(ifMatch, 'If-Match');
      const controller = new AbortController();
      let timer, rejectCancelled, cancelled, abortCause = null;
      const cancelledPromise = new Promise((_, reject) => { rejectCancelled = reject; });
      cancelled = () => { abortCause = 'CANCELLED'; controller.abort(); rejectCancelled(new ApiError('CANCELLED')); };
      signal?.addEventListener('abort', cancelled, {once: true});
      if (signal?.aborted) cancelled();
      const timeoutPromise = new Promise((_, reject) => {
        timer = setTimeout(() => { abortCause = 'TIMEOUT'; controller.abort(); reject(new ApiError('TIMEOUT')); }, timeoutMs);
      });
      try {
        const work = Promise.resolve().then(async () => {
          const headers = {'Accept': 'application/json'};
          if (body !== undefined) headers['Content-Type'] = 'application/json';
          if (idempotencyKey !== undefined) headers['Idempotency-Key'] = idempotencyKey;
          if (ifMatch !== undefined) headers['If-Match'] = ifMatch;
          if (actor) {
            headers['X-Touchline-Mock-Actor-Id'] = actor.id;
            headers['X-Touchline-Mock-Role'] = actor.role;
            headers['X-Touchline-Mock-School-Id'] = actor.schoolId;
          } else {
            const token = await config.getToken();
            if (typeof token !== 'string' || !token.trim()) throw new ApiError('UNAUTHENTICATED', 401);
            headers.Authorization = `Bearer ${token}`;
          }
          if (controller.signal.aborted) throw new ApiError(abortCause || 'CANCELLED');
          const response = await fetchImpl(`${baseUrl}${path}`, {
            method, headers, body: body === undefined ? undefined : JSON.stringify(body),
            signal: controller.signal, credentials: 'omit',
          });
          if (!response || typeof response.status !== 'number' || typeof response.json !== 'function') throw new ApiError('INVALID_RESPONSE');
          let payload;
          try { payload = await response.json(); }
          catch { throw new ApiError(abortCause || (response.status === 401 ? 'UNAUTHENTICATED' : 'INVALID_RESPONSE'), response.status); }
          if (!response.ok) {
            const issue = payload?.error;
            const fallback = response.status === 401 ? 'UNAUTHENTICATED' : response.status === 403 ? 'FORBIDDEN_SCOPE' :
              response.status === 404 ? 'NOT_FOUND' : response.status === 412 ? 'PRECONDITION_FAILED' : 'HTTP_ERROR';
            throw new ApiError(typeof issue?.code === 'string' ? issue.code : fallback, response.status,
              typeof issue?.target === 'string' ? issue.target : null);
          }
          if (!payload || typeof payload !== 'object' || Array.isArray(payload) || payload.error) throw new ApiError('INVALID_RESPONSE', response.status);
          return payload;
        });
        return await Promise.race([work, timeoutPromise, cancelledPromise]);
      } catch (error) {
        if (error instanceof ApiError) throw error;
        if (abortCause) throw new ApiError(abortCause);
        throw new ApiError('NETWORK_ERROR');
      } finally {
        clearTimeout(timer); signal?.removeEventListener('abort', cancelled);
      }
    }
    const api = '/api/v1';
    return Object.freeze({
      mode, mocked: !!config.mockFetch,
      reviewAssociation(schoolId, data, options = {}) {
        object(data, 'association'); natural(data.local_track_id, 'local_track_id'); natural(data.evidence_pts, 'evidence_pts');
        if (data.local_track_id === 0) throw new TypeError('local_track_id must be positive');
        required(data.run_id, 'run_id');
        if (!['confirmed', 'rejected', 'unknown', 'revoked'].includes(data.decision)) throw new TypeError('Invalid association decision');
        if (data.decision === 'confirmed') required(data.canonical_player_id, 'canonical_player_id');
        return request(`${api}/schools/${id(schoolId, 'school_id')}/association-reviews`, {...options, method: 'POST', body: data})
          .then(value => {
            result(value, ['review_id', 'school_id', 'run_id', 'decision', 'reviewed_by_user_id']);
            if (value.school_id !== schoolId || !Number.isSafeInteger(value.revision) || value.revision < 1 ||
              value.run_id !== data.run_id || value.local_track_id !== data.local_track_id || value.decision !== data.decision ||
              (data.decision === 'confirmed' && value.canonical_player_id !== data.canonical_player_id)) throw new ApiError('INVALID_RESPONSE');
            return value;
          });
      },
      createCorrection(schoolId, playerId, data, options = {}) {
        object(data, 'correction'); required(data.originating_session_id, 'originating_session_id');
        required(data.objective_id, 'objective_id'); required(data.observable_criterion, 'observable_criterion');
        required(data.initial_instruction, 'initial_instruction');
        return request(`${api}/schools/${id(schoolId, 'school_id')}/players/${id(playerId, 'player_id')}/corrections`, {...options, method: 'POST', body: data})
          .then(value => result(value, ['correction_id']));
      },
      attachEvidence(schoolId, correctionId, data, options = {}) {
        object(data, 'evidence');
        for (const field of ['session_id', 'run_id', 'association_review_id', 'media_id']) required(data[field], field);
        for (const field of ['time_base_num', 'time_base_den', 'start_pts', 'end_pts']) natural(data[field], field);
        if (data.time_base_num === 0 || data.time_base_den === 0 || data.end_pts <= data.start_pts) throw new TypeError('Invalid source time span');
        if (!Array.isArray(data.source_bbox_xyxy) || data.source_bbox_xyxy.length !== 4 ||
          !data.source_bbox_xyxy.every(Number.isFinite) || data.source_bbox_xyxy[2] <= data.source_bbox_xyxy[0] ||
          data.source_bbox_xyxy[3] <= data.source_bbox_xyxy[1]) throw new TypeError('Valid source bbox_xyxy is required');
        if (Object.hasOwn(data, 'review_status')) throw new TypeError('Evidence review status is server-controlled until OpenAPI confirms a review action');
        return request(`${api}/schools/${id(schoolId, 'school_id')}/corrections/${id(correctionId, 'correction_id')}/evidence`, {...options, method: 'POST', body: data})
          .then(value => result(value, ['evidence_id']));
      },
      submitAssessment(schoolId, data, options = {}) {
        object(data, 'assessment'); required(data.canonical_player_id, 'canonical_player_id'); required(data.correction_id, 'correction_id');
        if (!Array.isArray(data.session_ids) || data.session_ids.length < 2 || new Set(data.session_ids).size !== data.session_ids.length || data.session_ids.some(x => typeof x !== 'string' || !x)) throw new TypeError('Two distinct session IDs are required');
        if (!Array.isArray(data.evidence_ids) || !data.evidence_ids.length || data.evidence_ids.some(x => typeof x !== 'string' || !x)) throw new TypeError('Evidence IDs are required');
        if (!['improvement_observed', 'still_needs_work', 'consistent_execution_observed', 'unable_to_assess', 'insufficient_evidence'].includes(data.rating)) throw new TypeError('Invalid assessment outcome');
        required(data.coach_rationale, 'coach_rationale'); natural(data.revision_precondition, 'revision_precondition');
        return request(`${api}/schools/${id(schoolId, 'school_id')}/reports/assessments`, {...options, method: 'POST', body: data})
          .then(value => result(value, ['assessment_id']));
      },
      approveReport(schoolId, reportId, data, options = {}) {
        object(data, 'approval'); required(data.assessment_id, 'assessment_id'); required(data.expected_content_hash, 'expected_content_hash');
        if (data.coach_signature_acknowledgment !== true) throw new TypeError('Explicit coach acknowledgment is required');
        revision(options.ifMatch);
        return request(`${api}/schools/${id(schoolId, 'school_id')}/reports/${id(reportId, 'report_id')}/approve`, {...options, method: 'POST', body: data})
          .then(value => {
            result(value, ['approval_id', 'report_id', 'status', 'content_hash', 'approved_by_user_id']);
            if (value.report_id !== reportId || value.status !== 'approved' || value.content_hash !== data.expected_content_hash ||
              !Number.isSafeInteger(value.revision) || value.revision < 1) throw new ApiError('INVALID_RESPONSE');
            return value;
          });
      },
      async getGuardianReport(reportId, options = {}) {
        const value = result(await request(`${api}/guardian/reports/${id(reportId, 'report_id')}`, {...options, method: 'GET'}), ['report_id']);
        if (value.report_id !== reportId || !value.player || !value.assessment || !Array.isArray(value.evidence_moments)) throw new ApiError('INVALID_RESPONSE');
        return value;
      },
    });
  }
  function mediaUrlExpired(clip, now = Date.now()) {
    const expires = Date.parse(clip?.expires_at_utc || '');
    return !Number.isFinite(expires) || now >= expires;
  }
  const exported = Object.freeze({createClient, ApiError, mediaUrlExpired});
  root.TouchlinePlayerDevelopmentApi = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
