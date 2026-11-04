'use strict';
const test=require('node:test');const assert=require('node:assert/strict');
const {project,audioToVideo}=require('./timeline.js');
test('merges legacy cues, local markers and proposed events in video time order',()=>{
  const data={transcript:{segments:[{cue_id:'t2',text:'Second',video_start_s:4,video_end_s:5,speaker:'A',audio_recording_id:'a1'}],manual_cues:[{cue_id:'c1',text:'First',video_start_s:1,video_end_s:2}]},events:[{event_id:'e1',label:'Verified event',timestamp_s:3,session_id:'s1'}]};
  const local=[{marker_id:'m1',label:'Coach marker',timestamp_s:2,session_id:'s1'}];
  const entries=project('s1',data,local);assert.deepEqual(entries.map(e=>e.id),['c1','m1','e1','t2']);assert.equal(entries[3].speaker_label,'A');assert.equal(entries[3].audio_recording_id,'a1');assert.equal(entries[1].local_only,true);
});
test('rejects invalid, cross-session and empty entries',()=>{
  const entries=project('s1',{events:[{event_id:'wrong',label:'Other',timestamp_s:1,session_id:'s2'},{label:'Negative',timestamp_s:-1},{label:'No time'},{label:'Audio-only offset',start_offset_s:1},{label:'',timestamp_s:1},{label:'Good',timestamp_s:0}]});assert.deepEqual(entries.map(e=>e.text),['Good']);
});
test('preserves temporary track references without promoting identity',()=>{
  const [e]=project('s1',{events:[{event_id:'e',label:'Track observation',timestamp_s:1,local_track_id:7}]});assert.equal(e.local_track_id,7);assert.equal(e.player_id,null);
});
test('maps audio offsets by explicit anchor and measured drift slope',()=>{assert.ok(Math.abs(audioToVideo(10,5,1.001)-15.01)<1e-9);assert.throws(()=>audioToVideo(1,0,0));});
