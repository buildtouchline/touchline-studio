'use strict';
// Browser-only demo drafts. This is not an authenticated player record.
(function (root) {
  const KEY='touchline.coach_draft.v1';
  function validMoment(value) {
    return value && typeof value.run_id==='string' && value.run_id.length>0 &&
      Number.isFinite(value.video_time_s) && value.video_time_s>=0;
  }
  function validDraft(value) {
    return value && value.schema_version==='touchline.correction_draft.v1' &&
      value.status==='local_draft' && typeof value.id==='string' && value.id.length>0 &&
      typeof value.player_id==='string' && value.player_id.length>0 &&
      typeof value.focus==='string' && value.focus.length>0 &&
      typeof value.observable_check==='string' && value.observable_check.length>0 &&
      validMoment(value.first) && (value.followup===null ||
        (validMoment(value.followup) && value.followup.run_id!==value.first.run_id)) &&
      (value.coach_assessment===null || typeof value.coach_assessment==='string');
  }
  function read(storage) {
    const raw=storage.getItem(KEY);
    if(raw===null)return null;
    const value=JSON.parse(raw);
    if(!validDraft(value))throw new TypeError('Saved local correction is invalid');
    return value;
  }
  function save(storage,draft) {
    if(!validDraft(draft))throw new TypeError('Invalid local correction');
    storage.setItem(KEY,JSON.stringify(draft));
  }
  function remove(storage) { storage.removeItem(KEY); }
  const api=Object.freeze({KEY,read,save,remove,validDraft});
  root.TouchlineDevelopmentStorage=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
