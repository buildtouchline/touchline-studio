'use strict';
// Coach-side draft only. A local ledger label is not a server-verified player review.
(function (root) {
  function validTime(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
  function text(value) { return typeof value === 'string' ? value.trim() : ''; }
  function confirmedPlayer(roster, playerId) {
    return Array.isArray(roster) && roster.some(row => row && row.confirmed_player_id != null && String(row.confirmed_player_id) === String(playerId) && text(row.reviewed_by) && Number.isInteger(row.association_revision) && row.association_revision > 0);
  }
  function start({ id, playerId, playerName, focus, observableCheck, runId, timeS, roster }) {
    if (!text(id) || playerId == null || !confirmedPlayer(roster, playerId) || !text(focus) || !text(observableCheck) || !text(runId) || !validTime(timeS)) throw new TypeError('A locally associated player, observable correction and first video moment are required');
    return {schema_version:'touchline.correction_draft.v1',id:text(id),player_id:String(playerId),player_name:text(playerName),focus:text(focus),observable_check:text(observableCheck),first:{run_id:text(runId),video_time_s:timeS},followup:null,coach_assessment:null,status:'local_draft'};
  }
  function addFollowup(draft, {runId,timeS,roster}) {
    if (!draft || draft.status !== 'local_draft' || !text(runId) || text(runId) === draft.first.run_id || !validTime(timeS) || !confirmedPlayer(roster,draft.player_id)) throw new TypeError('A different run with the same local player label is required; identity remains unverified');
    return {...draft,followup:{run_id:text(runId),video_time_s:timeS},coach_assessment:null};
  }
  function assess(draft, assessment, outcome) {
    if (!draft?.followup || !text(assessment) || !['achieved','partially_achieved','still_a_focus','insufficient_evidence'].includes(outcome)) throw new TypeError('Two evidence moments, a coach outcome and an assessment are required');
    return {...draft,coach_assessment:text(assessment),coach_outcome:outcome};
  }
  const api=Object.freeze({start,addFollowup,assess,confirmedPlayer});
  root.TouchlineDevelopment=api;
  if (typeof module !== 'undefined' && module.exports) module.exports=api;
})(typeof window !== 'undefined' ? window : globalThis);
