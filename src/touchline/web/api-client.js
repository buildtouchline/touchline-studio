'use strict';
// UI-facing transport boundary. Production paths are supplied by a reviewed backend contract.
(function (root) {
  const STATES = new Set(['queued', 'running', 'processing', 'partial', 'succeeded', 'completed', 'complete', 'failed', 'cancelled', 'unavailable']);
  class ApiError extends Error {
    constructor(code, message, status = 0) {
      super(message || 'Touchline API is unavailable.');
      this.name = 'ApiError'; this.code = code; this.status = status;
    }
  }
  function record(value, label) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ApiError('INVALID_RESPONSE', `Invalid ${label} response.`);
    return value;
  }
  function list(value, label) {
    if (Array.isArray(value)) return value;
    if (Array.isArray(value?.items)) return value.items;
    throw new ApiError('INVALID_RESPONSE', `Invalid ${label} list.`);
  }
  function namedEntity(value, idField, nameField) {
    record(value, 'roster item');
    const id = value.id || value[idField];
    if (typeof id !== 'string' || !id) throw new ApiError('INVALID_RESPONSE', 'Roster ID is missing.');
    return {...value, id, name:typeof value.name === 'string' && value.name ? value.name : typeof value[nameField] === 'string' ? value[nameField] : id};
  }
  function status(value) {
    const raw = String(value || 'unavailable').toLowerCase();
    if (!STATES.has(raw)) throw new ApiError('INVALID_RESPONSE', 'Unknown analysis status.');
    return ({succeeded:'completed', complete:'completed', running:'processing'})[raw] || raw;
  }
  function job(value) {
    record(value, 'job');
    if (typeof value.job_id !== 'string' || !value.job_id) throw new ApiError('INVALID_RESPONSE', 'Job ID is missing.');
    const events = value.events === undefined ? [] : list(value.events, 'progress events');
    return {...value, status:status(value.status), events};
  }
  const SHA256 = /^[0-9a-f]{64}$/i;
  const verifiedHash = (...values) => {
    const value = values.find(item => typeof item === 'string' && SHA256.test(item));
    return value ? value.toLowerCase() : null;
  };
  function canonicalObservation(value, runId) {
    const frame = value?.frame;
    return value?.schema_version === 'touchline.observation.v1' &&
      typeof value.run_id === 'string' && value.run_id === runId &&
      frame && Number.isFinite(frame.timestamp_s) && frame.timestamp_s >= 0 &&
      Number.isInteger(frame.index) && frame.index >= 0 &&
      Array.isArray(value.bbox_xyxy) && value.bbox_xyxy.length === 4 &&
      value.bbox_xyxy.every(Number.isFinite) &&
      Number.isInteger(value.local_track_id) && value.local_track_id >= 0 &&
      (value.player_id === null || typeof value.player_id === 'string');
  }
  function sourceTimeFromFrameMap(rows, outputFrameIndex) {
    if (!Array.isArray(rows) || !Number.isInteger(outputFrameIndex)) return null;
    const row = rows.find(item => item?.output_frame_index === outputFrameIndex);
    const fraction = row?.source_relative_time;
    if (!Array.isArray(fraction) || fraction.length !== 2 ||
        !Number.isInteger(fraction[0]) || !Number.isInteger(fraction[1]) || fraction[1] <= 0) return null;
    const seconds = fraction[0] / fraction[1];
    return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
  }
  function sourceTimeFromPlayback(rows, playbackTimeS) {
    if(!Array.isArray(rows)||!Number.isFinite(playbackTimeS)||playbackTimeS<0)return null;
    const outputTime=row=>{
      const base=row?.output_time_base;
      if(!Array.isArray(base)||base.length!==2||!base.every(Number.isInteger)||base[0]<=0||base[1]<=0||!Number.isInteger(row.output_pts))return null;
      return row.output_pts*base[0]/base[1];
    };
    let bestIndex=-1,bestDelta=Infinity,bestTime=null;
    for(let index=0;index<rows.length;index++){
      const time=outputTime(rows[index]);if(time===null)continue;
      const delta=Math.abs(time-playbackTimeS);
      if(delta<bestDelta){bestIndex=index;bestDelta=delta;bestTime=time;}
    }
    if(bestIndex<0)return null;
    const neighbours=[outputTime(rows[bestIndex-1]),outputTime(rows[bestIndex+1])].filter(time=>time!==null);
    const spacing=Math.min(...neighbours.map(time=>Math.abs(time-bestTime)).filter(value=>value>0));
    if(!Number.isFinite(spacing)||bestDelta>spacing*0.55+1e-6)return null;
    return sourceTimeFromFrameMap(rows,rows[bestIndex].output_frame_index);
  }
  function reviewSummary(value) {
    const manifest = value?.manifest || {};
    const observations = Array.isArray(value?.observations) ? value.observations : [];
    const runId = typeof value?.run_id === 'string' ? value.run_id : manifest.run_id;
    return {
      runId:typeof runId === 'string' ? runId : null,
      status:typeof manifest.status === 'string' && STATES.has(manifest.status.toLowerCase()) ? status(manifest.status) : 'unavailable',
      sourceHash:verifiedHash(manifest.source?.sha256,manifest.source_sha256,value?.normalization?.source_sha256),
      normalizedHash:verifiedHash(manifest.normalized_video_sha256,value?.normalization?.video_sha256),
      parentRunId:typeof manifest.parent_run_id === 'string' ? manifest.parent_run_id : null,
      observationCount:observations.length,
      compatibleObservations:observations.length > 0 && observations.every(item => canonicalObservation(item, runId)),
      frameMapAvailable:Array.isArray(value?.frame_map) && value.frame_map.length > 0,
      annotatedVideoAvailable:typeof value?.annotated_video_url === 'string' &&
        value.annotated_video_url.startsWith('/') && !value.annotated_video_url.startsWith('//') &&
        value.annotated_video_verified === true,
      correctionPersistenceAvailable:false,
    };
  }
  function path(template, params = {}) {
    if (typeof template !== 'string' || !template.startsWith('/') || template.startsWith('//')) throw new ApiError('CONFIGURATION_REQUIRED', 'A reviewed backend route is required.');
    const value = template.replace(/\{([A-Za-z][A-Za-z0-9_]*)\}/g, (_, key) => {
      const item = params[key];
      if (typeof item !== 'string' || !item) throw new TypeError(`${key} is required`);
      return encodeURIComponent(item);
    });
    if (value.includes('{') || value.includes('}')) throw new ApiError('CONFIGURATION_REQUIRED', 'Backend route parameters are incomplete.');
    return value;
  }
  function createClient(config = {}) {
    const mode = config.mode || 'local';
    if (!['demo', 'local', 'production'].includes(mode)) throw new TypeError('Unknown Touchline API mode');
    if (mode === 'demo' && config.syntheticFixture !== true) throw new ApiError('CONFIGURATION_REQUIRED', 'Synthetic demo mode requires an explicit fixture context.');
    const routes = config.routes || {};
    const suppliedBaseUrl = config.baseUrl || '';
    if (typeof suppliedBaseUrl !== 'string') throw new TypeError('API base URL must be a string');
    const baseUrl = suppliedBaseUrl.replace(/\/$/, '');
    if (baseUrl) {
      let parsed;
      try { parsed = new URL(baseUrl); }
      catch { throw new TypeError('API base URL is invalid'); }
      if (parsed.username || parsed.password || parsed.search || parsed.hash || parsed.pathname !== '/' ||
          (parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && ['localhost','127.0.0.1','[::1]'].includes(parsed.hostname)))) throw new TypeError('API base URL must be an HTTPS origin or loopback HTTP origin');
    }
    if (mode === 'production' && typeof config.getToken !== 'function') throw new ApiError('CONFIGURATION_REQUIRED', 'Production authentication is not configured.');
    const fetchImpl = config.fetchImpl || root.fetch?.bind(root);
    if (typeof fetchImpl !== 'function') throw new TypeError('Fetch transport is required');
    const timeoutMs = config.timeoutMs == null ? 12000 : config.timeoutMs;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new TypeError('API timeout must be positive');
    async function request(route, {method = 'GET', body, signal, headers = {}} = {}) {
      if (signal?.aborted) throw new ApiError('CANCELLED', 'Request cancelled.');
      const token = mode === 'production' ? await config.getToken() : null;
      if (mode === 'production' && (typeof token !== 'string' || !token.trim())) throw new ApiError('AUTHENTICATION_REQUIRED', 'Sign in to access Touchline.', 401);
      const requestHeaders = {'Accept':'application/json', ...headers};
      if (body !== undefined && !(body instanceof Blob)) requestHeaders['Content-Type'] = 'application/json';
      if (token) requestHeaders.Authorization = `Bearer ${token}`;
      const controller = new AbortController();
      let rejectCancelled;
      const cancelled = new Promise((_, reject) => { rejectCancelled = reject; });
      const cancel = () => { controller.abort(); rejectCancelled(new ApiError('CANCELLED', 'Request cancelled.')); };
      signal?.addEventListener('abort', cancel, {once:true});
      if (signal?.aborted) cancel();
      let timer;
      const timeout = new Promise((_, reject) => { timer = setTimeout(() => { controller.abort(); reject(new ApiError('TIMEOUT', 'Touchline API timed out. Try again.')); }, timeoutMs); });
      try {
        const work = async () => {
          const response = await fetchImpl(`${baseUrl}${route}`, {method, headers:requestHeaders, body:body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body), signal:controller.signal, credentials:'omit'});
          let value;
          try { value = await response.json(); }
          catch { throw new ApiError('INVALID_RESPONSE', `Touchline API returned HTTP ${response.status} without JSON.`, response.status); }
          if (!response.ok || value?.error) {
            const code = typeof value?.error?.code === 'string' ? value.error.code : response.status === 401 ? 'AUTHENTICATION_REQUIRED' : response.status === 403 ? 'FORBIDDEN' : 'API_ERROR';
            throw new ApiError(code, code === 'AUTHENTICATION_REQUIRED' ? 'Sign in to access Touchline.' : `Touchline API returned HTTP ${response.status}.`, response.status);
          }
          return value;
        };
        return await Promise.race([work(), timeout, cancelled]);
      } catch (error) {
        if (error instanceof ApiError) throw error;
        if (signal?.aborted) throw new ApiError('CANCELLED', 'Request cancelled.');
        if (controller.signal.aborted) throw new ApiError('TIMEOUT', 'Touchline API timed out. Try again.');
        throw new ApiError('NETWORK_ERROR', 'Touchline API could not be reached.');
      } finally {
        clearTimeout(timer);
        signal?.removeEventListener('abort', cancel);
      }
    }
    function productionRoute(name, params) {
      if (mode !== 'production') throw new ApiError('UNAVAILABLE', 'This action needs the authenticated backend.');
      return path(routes[name], params);
    }
    const api = {
      mode,
      async readiness() {
        if (mode === 'demo') return {ready:true, simulated:true, message:'Synthetic upload simulation only. No video is analysed.'};
        if (mode === 'local') return record(await request('/api/analyze/readiness'), 'readiness');
        if (!routes.readiness) return {ready:false, code:'CONFIGURATION_REQUIRED', message:'Backend recording readiness route must be published and configured before upload.'};
        const value = record(await request(productionRoute('readiness')), 'readiness');
        if (typeof value.ready !== 'boolean') throw new ApiError('INVALID_RESPONSE', 'Backend readiness state is missing.');
        return value;
      },
      async getSessions(options = {}) {
        return list(await request(mode === 'production' ? productionRoute('getSessions') : '/api/runs', options), 'sessions').map(item => {
          record(item, 'session');
          if (typeof item.id !== 'string' || !item.id) throw new ApiError('INVALID_RESPONSE', 'Session ID is missing.');
          return item;
        });
      },
      async getRun(id, options = {}) {
        const value = record(await request(mode === 'production' ? productionRoute('getRun', {id}) : `/api/run/${encodeURIComponent(id)}`, options), 'run');
        if (mode === 'production' && (!value.manifest || !Array.isArray(value.observations) || !value.analytics || typeof value.analytics !== 'object' ||
            value.observations.some(item => !canonicalObservation(item, value.run_id || value.manifest.run_id)))) throw new ApiError('INVALID_RESPONSE', 'Backend run contract does not match the review workspace.');
        return value;
      },
      async getObservationPage(id, offset, limit = 5000, options = {}) {
        if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 5000)
          throw new TypeError('A bounded observation page is required.');
        if (mode === 'production' && !routes.getObservationPage)
          throw new ApiError('UNAVAILABLE', 'The backend has not published an observation paging contract.');
        const route = mode === 'production'
          ? productionRoute('getObservationPage', {id})
          : `/api/run/${encodeURIComponent(id)}/observations`;
        const separator = route.includes('?') ? '&' : '?';
        const page = record(await request(`${route}${separator}offset=${offset}&limit=${limit}`, options), 'observation page');
        if (!Array.isArray(page.observations) ||
            (page.next_offset !== null && page.next_offset !== undefined &&
             (!Number.isSafeInteger(page.next_offset) || page.next_offset <= offset)))
          throw new ApiError('INVALID_RESPONSE', 'Invalid observation page cursor.');
        if (mode === 'production' && page.observations.some(item => !canonicalObservation(item, id)))
          throw new ApiError('INVALID_RESPONSE', 'Backend observation page does not match the review schema.');
        return page;
      },
      async getJob(id, options = {}) {
        const value = await request(mode === 'production' ? productionRoute('getJob', {id}) : `/api/job/${encodeURIComponent(id)}`, options);
        if (mode !== 'production' && value?.status === 'not_found') return job({job_id:id, status:'unavailable'});
        return job(value);
      },
      async getJobProgress(id, options = {}) {
        if (mode !== 'production' || !routes.getJobProgress) return api.getJob(id, options);
        return job(await request(productionRoute('getJobProgress', {id}), options));
      },
      async getAnalysisResults(id, options = {}) { return record(await request(productionRoute('getAnalysisResults', {id}), options), 'analysis results'); },
      async getTeams(options = {}) { return list(await request(productionRoute('getTeams'), options), 'teams').map(item => namedEntity(item, 'team_id', 'team_name')); },
      async getPlayers(teamId, options = {}) { return list(await request(productionRoute('getPlayers', {teamId}), options), 'players').map(item => namedEntity(item, 'player_id', 'player_name')); },
      async createSession(data, options = {}) { return record(await request(productionRoute('createSession'), {...options, method:'POST', body:record(data, 'session')}), 'session'); },
      async createUpload(data, options = {}) { return record(await request(productionRoute('createUpload'), {...options, method:'POST', body:record(data, 'upload')}), 'upload'); },
      async uploadRecording(uploadId, file, options = {}) {
        if (!(file instanceof Blob) || !file.size) throw new TypeError('A nonempty recording is required');
        return record(await request(productionRoute('uploadRecording', {uploadId}), {...options, method:'PUT', body:file, headers:{'Content-Type':file.type || 'application/octet-stream'}}), 'upload');
      },
      async finalizeUpload(uploadId, data, options = {}) { return record(await request(productionRoute('finalizeUpload', {uploadId}), {...options, method:'POST', body:record(data, 'upload finalization')}), 'upload'); },
      async createAnalysisJob(data, options = {}) { return job(await request(productionRoute('createAnalysisJob'), {...options, method:'POST', body:record(data, 'analysis job')})); },
      async getInstructions(playerId, options = {}) { return list(await request(productionRoute('getInstructions', {playerId}), options), 'instructions'); },
      async saveInstruction(playerId, data, options = {}) { return record(await request(productionRoute('saveInstruction', {playerId}), {...options, method:'POST', body:record(data, 'instruction')}), 'instruction'); },
      async getFeedback(playerId, options = {}) { return list(await request(productionRoute('getFeedback', {playerId}), options), 'feedback'); },
      async saveFeedback(playerId, data, options = {}) { return record(await request(productionRoute('saveFeedback', {playerId}), {...options, method:'POST', body:record(data, 'feedback')}), 'feedback'); },
      async getReports(playerId, options = {}) { return list(await request(productionRoute('getReports', {playerId}), options), 'reports'); },
      async generateReport(playerId, data, options = {}) { return record(await request(productionRoute('generateReport', {playerId}), {...options, method:'POST', body:record(data, 'report request')}), 'report'); },
    };
    return Object.freeze(api);
  }
  const exported = Object.freeze({createClient, ApiError, status, job, canonicalObservation, sourceTimeFromFrameMap, sourceTimeFromPlayback, reviewSummary});
  root.TouchlineApi = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
