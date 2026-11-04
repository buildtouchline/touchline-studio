'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {createClient, ApiError, mediaUrlExpired} = require('./player-development-api.js');
const {createSyntheticFixtureTransport} = require('./player-development-mock.js');
const origin = 'http://127.0.0.1:8000';
const actor = {id: 'synthetic_coach_1', role: 'coach', schoolId: 'synthetic_sch_1'};
const guardian = {id: 'synthetic_parent_1', role: 'guardian', schoolId: 'synthetic_sch_1'};
const association = {run_id: 'synthetic_run_A', local_track_id: 7, canonical_player_id: 'synthetic_player_1', decision: 'confirmed', evidence_pts: 90000, prior_decision_id: null, coach_notes: 'Reviewed video'};
const correction = {originating_session_id: 'synthetic_session_A', objective_id: 'scan', observable_criterion: 'Look before receiving', initial_instruction: 'Check shoulder', status: 'active'};
const evidence = {session_id: 'synthetic_session_B', run_id: 'synthetic_run_B', association_review_id: 'synthetic_asv_2', media_id: 'synthetic_media_B', time_base_num: 1, time_base_den: 90000, start_pts: 147600, end_pts: 154800, source_bbox_xyxy: [3005, 262, 3026, 307], coach_annotation: 'Moment reviewed'};
const assessment = {canonical_player_id: 'synthetic_player_1', correction_id: 'synthetic_cor_1', session_ids: ['synthetic_session_A', 'synthetic_session_B'], evidence_ids: ['synthetic_evi_1', 'synthetic_evi_2'], rating: 'still_needs_work', coach_rationale: 'Selected moments remain limited', revision_precondition: 1};
const approval = {assessment_id: 'synthetic_asm_1', expected_content_hash: 'sha256_synthetic_fixture', coach_signature_acknowledgment: true};
const local = (transport, extra = {}) => createClient({mode: 'local-test', baseUrl: origin, mockActor: actor, mockFetch: transport.fetchMock, ...extra});

test('six provisional routes preserve method, body, idempotency and revision headers', async () => {
  const fixture = createSyntheticFixtureTransport(); const client = local(fixture);
  const first = await client.reviewAssociation(actor.schoolId, association, {idempotencyKey: 'association-key'});
  assert.equal(first.review_id, 'synthetic_asv_synthetic_run_A_7');
  assert.equal((await client.createCorrection(actor.schoolId, association.canonical_player_id, correction, {idempotencyKey: 'correction-key'})).correction_id, 'synthetic_cor_1');
  assert.equal((await client.attachEvidence(actor.schoolId, 'synthetic_cor_1', evidence, {idempotencyKey: 'evidence-key'})).review_status, 'pending');
  assert.equal((await client.submitAssessment(actor.schoolId, assessment, {idempotencyKey: 'assessment-key'})).report_id, 'synthetic_rep_1');
  assert.equal((await client.approveReport(actor.schoolId, 'synthetic_rep_1', approval, {ifMatch: '"rev_1"', idempotencyKey: 'approval-key'})).status, 'approved');
  await assert.rejects(client.getGuardianReport('synthetic_rep_1'), e => e.code === 'FORBIDDEN_SCOPE');
  assert.equal((await local(fixture, {mockActor: guardian}).getGuardianReport('synthetic_rep_1')).player.display_name, 'Synthetic player');
  assert.deepEqual(fixture.calls.filter(x => x.method !== 'GET').map(x => `${x.method} ${x.path}`), [
    'POST /api/v1/schools/synthetic_sch_1/association-reviews',
    'POST /api/v1/schools/synthetic_sch_1/players/synthetic_player_1/corrections',
    'POST /api/v1/schools/synthetic_sch_1/corrections/synthetic_cor_1/evidence',
    'POST /api/v1/schools/synthetic_sch_1/reports/assessments',
    'POST /api/v1/schools/synthetic_sch_1/reports/synthetic_rep_1/approve',
  ]);
  assert.equal(fixture.calls.at(-1).headers['X-Touchline-Mock-Role'], 'guardian');
  assert.equal(fixture.calls[0].headers['Idempotency-Key'], 'association-key');
  assert.equal(fixture.calls[4].headers['If-Match'], '"rev_1"');
  assert.equal(fixture.calls[0].headers['X-Touchline-Mock-Actor-Id'], actor.id);
  assert.equal(fixture.calls[0].headers.Authorization, undefined);
  assert.deepEqual(JSON.parse(fixture.calls[2].body), evidence);
  assert.match(fixture.label, /not saved to backend/);
});

test('mock actor and transport cannot activate in production or at a remote host', () => {
  const fixture = createSyntheticFixtureTransport();
  assert.throws(() => createClient({baseUrl: origin, mockActor: actor, getToken: () => 'token'}), /Mock actor/);
  assert.throws(() => createClient({mode: 'local-test', baseUrl: 'https://service.example', mockActor: actor, mockFetch: fixture.fetchMock}), /loopback/);
  assert.throws(() => createClient({baseUrl: origin, getToken: () => 'token', mockFetch: fixture.fetchMock}), /Mock transport/);
  assert.throws(() => createClient({baseUrl: 'http://service.example', getToken: () => 'token'}), /HTTPS/);
});

test('production sends only bearer authentication and handles missing token', async () => {
  let seen;
  const fetchImpl = async (_url, options) => { seen = options; return {ok: true, status: 200, json: async () => ({report_id: 'r1', player: {}, assessment: {}, evidence_moments: []})}; };
  const client = createClient({baseUrl: 'https://touchline.example', getToken: async () => 'secret-token', fetchImpl});
  await client.getGuardianReport('r1');
  assert.equal(seen.headers.Authorization, 'Bearer secret-token');
  assert.equal(seen.headers['X-Touchline-Mock-Role'], undefined);
  await assert.rejects(createClient({getToken: () => '', fetchImpl}).getGuardianReport('r1'), e => e instanceof ApiError && e.code === 'UNAUTHENTICATED');
});

test('known errors are safe and preserve code, status and target', async () => {
  for (const [status, code] of [[401, 'UNAUTHENTICATED'], [403, 'FORBIDDEN_SCOPE'], [403, 'CONSENT_REVOKED'], [412, 'PRECONDITION_FAILED'], [412, 'ASSOCIATION_MUTATED'], [422, 'ASSOCIATION_UNCONFIRMED'], [422, 'CROSS_SESSION_MISMATCH'], [422, 'MEDIA_OUT_OF_BOUNDS']]) {
    const path = 'GET /api/v1/guardian/reports/synthetic_rep_1';
    const fixture = createSyntheticFixtureTransport({failures: {[path]: {status, code}}});
    await assert.rejects(local(fixture, {mockActor: guardian}).getGuardianReport('synthetic_rep_1'), e => e instanceof ApiError && e.status === status && e.code === code && !e.message.includes('Synthetic fixture failure'));
  }
});

test('two sessions require separate review responses; failed writes leave coach input intact', async () => {
  const fixture = createSyntheticFixtureTransport(); const client = local(fixture);
  const first = await client.reviewAssociation(actor.schoolId, association);
  const second = await client.reviewAssociation(actor.schoolId, {...association, run_id: 'synthetic_run_B', local_track_id: 12, evidence_pts: 147600});
  assert.notEqual(first.review_id, second.review_id);
  assert.equal(first.canonical_player_id, second.canonical_player_id);
  assert.notEqual(first.run_id, second.run_id);
  // Matching snapshots are only candidate continuity; no current-review GET exists to prove either remains valid.
  const path = 'POST /api/v1/schools/synthetic_sch_1/reports/assessments';
  const conflict = local(createSyntheticFixtureTransport({failures: {[path]: {status: 412, code: 'ASSOCIATION_MUTATED'}}}));
  const unsaved = structuredClone(assessment);
  await assert.rejects(conflict.submitAssessment(actor.schoolId, unsaved, {idempotencyKey: 'same-retry-key'}), e => e.code === 'ASSOCIATION_MUTATED');
  assert.deepEqual(unsaved, assessment);
});

test('timeout and external cancellation settle even when transport ignores abort', async () => {
  const stuck = () => new Promise(() => {});
  const timed = createClient({mode: 'local-test', baseUrl: origin, mockActor: actor, mockFetch: stuck, timeoutMs: 10});
  await assert.rejects(timed.getGuardianReport('synthetic_rep_1'), e => e.code === 'TIMEOUT');
  const controller = new AbortController();
  const cancelled = timed.getGuardianReport('synthetic_rep_1', {signal: controller.signal});
  controller.abort();
  await assert.rejects(cancelled, e => e.code === 'CANCELLED');
});

test('abort-aware transport still reports timeout and cancellation accurately', async () => {
  const abortAware = (_url, options) => new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(new Error('transport aborted')), {once: true}));
  const client = createClient({mode: 'local-test', baseUrl: origin, mockActor: actor, mockFetch: abortAware, timeoutMs: 10});
  await assert.rejects(client.getGuardianReport('synthetic_rep_1'), e => e.code === 'TIMEOUT');
  const controller = new AbortController();
  const pending = client.getGuardianReport('synthetic_rep_1', {signal: controller.signal});
  controller.abort();
  await assert.rejects(pending, e => e.code === 'CANCELLED');
});

test('malformed success, mismatched approval and malformed 401 fail closed', async () => {
  const broken = createClient({mode: 'local-test', baseUrl: origin, mockActor: actor, mockFetch: async () => ({ok: true, status: 200, json: async () => ({report_id: 'wrong'})})});
  await assert.rejects(broken.getGuardianReport('synthetic_rep_1'), e => e.code === 'INVALID_RESPONSE');
  const wrongApproval = createClient({mode: 'local-test', baseUrl: origin, mockActor: actor, mockFetch: async () => ({ok: true, status: 200, json: async () => ({approval_id: 'x', report_id: 'wrong', status: 'approved', content_hash: approval.expected_content_hash})})});
  await assert.rejects(wrongApproval.approveReport(actor.schoolId, 'synthetic_rep_1', approval, {ifMatch: '"rev_1"'}), e => e.code === 'INVALID_RESPONSE');
  const unauthorized = createClient({mode: 'local-test', baseUrl: origin, mockActor: actor, mockFetch: async () => ({ok: false, status: 401, json: async () => { throw new SyntaxError('bad JSON'); }})});
  await assert.rejects(unauthorized.getGuardianReport('synthetic_rep_1'), e => e.code === 'UNAUTHENTICATED');
});

test('missing source geometry, timebase, review ID and client-set review status are rejected before network', async () => {
  const fixture = createSyntheticFixtureTransport(); const client = local(fixture);
  for (const change of [{association_review_id: ''}, {time_base_den: 0}, {start_pts: 160000}, {source_bbox_xyxy: null}, {review_status: 'accepted'}]) {
    assert.throws(() => client.attachEvidence(actor.schoolId, 'synthetic_cor_1', {...evidence, ...change}), TypeError);
  }
  assert.equal(fixture.calls.length, 0);
});

test('assessment requires distinct sessions and one of five outcomes; approval requires precondition', async () => {
  const client = local(createSyntheticFixtureTransport());
  for (const rating of ['improvement_observed', 'still_needs_work', 'consistent_execution_observed', 'unable_to_assess', 'insufficient_evidence']) {
    assert.equal((await client.submitAssessment(actor.schoolId, {...assessment, rating})).assessment_id, 'synthetic_asm_1');
  }
  assert.throws(() => client.submitAssessment(actor.schoolId, {...assessment, session_ids: ['same', 'same']}), TypeError);
  assert.throws(() => client.approveReport(actor.schoolId, 'r', approval), TypeError);
  assert.throws(() => client.approveReport(actor.schoolId, 'r', {...approval, coach_signature_acknowledgment: false}, {ifMatch: '"rev_1"'}), TypeError);
});

test('media URL expiry is explicit and missing expiry is unsafe', () => {
  assert.equal(mediaUrlExpired({expires_at_utc: '2026-10-31T01:00:00Z'}, Date.parse('2026-10-31T00:59:59Z')), false);
  assert.equal(mediaUrlExpired({expires_at_utc: '2026-10-31T01:00:00Z'}, Date.parse('2026-10-31T01:00:00Z')), true);
  assert.equal(mediaUrlExpired({url: 'https://media.example/clip'}), true);
});
