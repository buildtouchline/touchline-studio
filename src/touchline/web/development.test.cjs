'use strict';
const test=require('node:test');const assert=require('node:assert/strict');
const {start,addFollowup,assess}=require('./development.js');
const roster=[{confirmed_player_id:12,local_track_id:8,player_name:'Player A',reviewed_by:'coach-1',association_revision:2},{confirmed_player_id:null,local_track_id:7}];
const first=()=>start({id:'d1',playerId:12,playerName:'Player A',focus:'Check shoulder before receiving',observableCheck:'Head turns before first touch',runId:'s1',timeS:2,roster});
test('requires an explicitly confirmed roster player and an observable check',()=>{
  assert.throws(()=>start({id:'d2',playerId:7,focus:'Check shoulder',observableCheck:'Head turns',runId:'s1',timeS:1,roster}));
  assert.throws(()=>start({id:'d2',playerId:12,focus:'Check shoulder',observableCheck:'Head turns',runId:'s1',timeS:1,roster:[{confirmed_player_id:12}]}));
  assert.throws(()=>start({id:'d2',playerId:12,focus:'Check shoulder',observableCheck:'',runId:'s1',timeS:1,roster}));
  assert.equal(first().status,'local_draft');
});
test('requires different sessions and the same confirmed player for follow-up',()=>{
  assert.throws(()=>addFollowup(first(),{runId:'s1',timeS:3,roster}));
  assert.throws(()=>addFollowup(first(),{runId:'s2',timeS:3,roster:[{confirmed_player_id:null}]}));
  assert.deepEqual(addFollowup(first(),{runId:'s2',timeS:3,roster}).followup,{run_id:'s2',video_time_s:3});
});
test('assessment requires both reviewed moments and never changes local draft status',()=>{
  assert.throws(()=>assess(first(),'Improved'));
  assert.throws(()=>assess(addFollowup(first(),{runId:'s2',timeS:3,roster}),'Head turn observed in selected clip','automatic_improvement'));
  const result=assess(addFollowup(first(),{runId:'s2',timeS:3,roster}),'Head turn observed in selected clip','partially_achieved');
  assert.equal(result.coach_assessment,'Head turn observed in selected clip');
  assert.equal(result.coach_outcome,'partially_achieved');
  assert.equal(result.status,'local_draft');
});
