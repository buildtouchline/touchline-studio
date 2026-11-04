'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {read,save,remove}=require('./development-storage.js');
const {start,addFollowup,assess}=require('./development.js');
const roster=[{confirmed_player_id:12,reviewed_by:'fixture-coach',association_revision:1}];
function storage(){const values=new Map();return {getItem:key=>values.has(key)?values.get(key):null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};}
test('browser-local draft survives a fresh read and preserves two distinct moments and coach outcome',()=>{
  const store=storage();
  const first=start({id:'draft-1',playerId:12,focus:'Check shoulder',observableCheck:'Look before receiving',runId:'one',timeS:1,roster});
  const followup=addFollowup(first,{runId:'two',timeS:2,roster});
  const reviewed=assess(followup,'Coach saw one head turn.','partially_achieved');
  save(store,reviewed);
  assert.deepEqual(read(store),reviewed);
  remove(store);
  assert.equal(read(store),null);
});
test('invalid saved value is rejected and not treated as a confirmed record',()=>{
  const store=storage();store.setItem('touchline.coach_draft.v1',JSON.stringify({status:'approved',player_id:'12'}));
  assert.throws(()=>read(store),/invalid/);
});
