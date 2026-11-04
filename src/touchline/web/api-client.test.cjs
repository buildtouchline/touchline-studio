'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {createClient, ApiError, job} = require('./api-client.js');

const response = (value, status = 200) => ({ok:status >= 200 && status < 300, status, json:async()=>value});

test('demo reads only the fixture transport and marks readiness as simulated', async () => {
  const calls=[];
  const api=createClient({mode:'demo',syntheticFixture:true,fetchImpl:async url=>{calls.push(url);return response([{id:'fixture-one',synthetic:true}]);}});
  assert.deepEqual(await api.getSessions(),[{id:'fixture-one',synthetic:true}]);
  assert.deepEqual(calls,['/api/runs']);
  assert.deepEqual(await api.readiness(),{ready:true,simulated:true,message:'Synthetic upload simulation only. No video is analysed.'});
  await assert.rejects(api.createAnalysisJob({upload_id:'u',client_request_id:'r'}),error=>error.code==='UNAVAILABLE');
});

test('local mode reads Studio routes and never treats them as production writes', async () => {
  const calls=[];
  const api=createClient({mode:'local',fetchImpl:async url=>{
    calls.push(url);
    if(url==='/api/runs')return response([{id:'run-a',status:'partial'}]);
    if(url==='/api/run/run-a')return response({manifest:{status:'partial'},observations:[],analytics:{}});
    return response({job_id:'run-a',status:'processing'});
  }});
  assert.equal((await api.getSessions())[0].status,'partial');
  assert.equal((await api.getRun('run-a')).manifest.status,'partial');
  assert.equal((await api.getJob('run-a')).status,'processing');
  assert.deepEqual(calls,['/api/runs','/api/run/run-a','/api/job/run-a']);
  await assert.rejects(api.getTeams(),error=>error.code==='UNAVAILABLE');
});

test('local observation pages use the published Studio cursor and reject stalled pages', async () => {
  const calls=[];
  const api=createClient({mode:'local',fetchImpl:async url=>{calls.push(url);return response({observations:[{frame:{index:1}}],next_offset:null});}});
  assert.equal((await api.getObservationPage('run-a',1200,5000)).observations.length,1);
  assert.deepEqual(calls,['/api/run/run-a/observations?offset=1200&limit=5000']);
  await assert.rejects(api.getObservationPage('run-a',-1),TypeError);
  const bad=createClient({mode:'local',fetchImpl:async()=>response({observations:[],next_offset:1200})});
  await assert.rejects(bad.getObservationPage('run-a',1200),error=>error.code==='INVALID_RESPONSE');
});

test('production cannot page observations without a published route or accept experimental rows', async () => {
  const noRoute=createClient({mode:'production',getToken:async()=> 'token',fetchImpl:async()=>{throw Error('must not fetch');}});
  await assert.rejects(noRoute.getObservationPage('run-a',0),error=>error.code==='UNAVAILABLE');
  const configured=createClient({mode:'production',routes:{getObservationPage:'/runs/{id}/observations'},getToken:async()=> 'token',
    fetchImpl:async()=>response({observations:[{frame_index:1,track_id:4}],next_offset:null})});
  await assert.rejects(configured.getObservationPage('run-a',0),error=>error.code==='INVALID_RESPONSE');
});

test('production requires authentication and reviewed routes, without fixture fallback', async () => {
  assert.throws(()=>createClient({mode:'demo'}),error=>error.code==='CONFIGURATION_REQUIRED');
  assert.throws(()=>createClient({mode:'production',baseUrl:'https://user:pass@api.example.test',getToken:async()=> 'token'}),TypeError);
  assert.throws(()=>createClient({mode:'production',baseUrl:'http://api.example.test',getToken:async()=> 'token'}),TypeError);
  assert.throws(()=>createClient({mode:'production'}),error=>error.code==='CONFIGURATION_REQUIRED');
  let calls=0;
  const api=createClient({mode:'production',getToken:async()=>'',fetchImpl:async()=>{calls++;return response([]);}});
  await assert.rejects(api.getSessions(),error=>error.code==='CONFIGURATION_REQUIRED');
  assert.equal(calls,0);
  const configured=createClient({mode:'production',routes:{getSessions:'/v1/sessions'},getToken:async()=>'',fetchImpl:async()=>{calls++;return response([]);}});
  await assert.rejects(configured.getSessions(),error=>error.code==='AUTHENTICATION_REQUIRED');
  assert.equal(calls,0);
});

test('production sends bearer only to configured HTTPS route and preserves failure state', async () => {
  let actual;
  const api=createClient({mode:'production',baseUrl:'https://api.example.test',routes:{getJob:'/jobs/{id}'},getToken:async()=> 'test-token',fetchImpl:async(url,options)=>{
    actual={url,options};return response({job_id:'job-1',attempt_id:'attempt-2',status:'running',stage:'normalizing',events:[{event:'saved_frames',frame_count:12}]});
  }});
  const state=await api.getJob('job-1');
  assert.equal(state.status,'processing');
  assert.equal(state.attempt_id,'attempt-2');
  assert.equal(state.events[0].frame_count,12);
  assert.equal(actual.url,'https://api.example.test/jobs/job-1');
  assert.equal(actual.options.headers.Authorization,'Bearer test-token');
  assert.equal(actual.options.credentials,'omit');
  assert.equal(actual.options.headers['X-Touchline-Mock-Actor-Id'],undefined);
});

test('missing production readiness remains closed even when upload routes are configured', async () => {
  const api=createClient({mode:'production',routes:{createUpload:'/uploads',uploadRecording:'/uploads/{uploadId}',finalizeUpload:'/uploads/{uploadId}/finalize',createAnalysisJob:'/jobs'},getToken:async()=> 'token',fetchImpl:async()=>{throw Error('must not call');}});
  assert.deepEqual((await api.readiness()).ready,false);
});

test('production failure never retries against Studio or demo fixtures', async () => {
  const calls=[];
  const api=createClient({mode:'production',baseUrl:'https://api.example.test',routes:{getSessions:'/authorized/sessions'},getToken:async()=> 'token',fetchImpl:async url=>{calls.push(url);return response({error:{code:'BACKEND_UNAVAILABLE'}},503);}});
  await assert.rejects(api.getSessions(),error=>error.code==='BACKEND_UNAVAILABLE'&&error.status===503);
  assert.deepEqual(calls,['https://api.example.test/authorized/sessions']);
});

test('production rejects a run shape that cannot support evidence review', async () => {
  const api=createClient({mode:'production',routes:{getRun:'/runs/{id}'},getToken:async()=> 'token',fetchImpl:async()=>response({status:'succeeded'})});
  await assert.rejects(api.getRun('run'),error=>error.code==='INVALID_RESPONSE');
});

test('production refuses experimental ResNet rows instead of presenting them as canonical evidence', async () => {
  const api=createClient({mode:'production',routes:{getRun:'/runs/{id}'},getToken:async()=> 'token',fetchImpl:async()=>response({
    run_id:'run-1',manifest:{run_id:'run-1',status:'complete'},analytics:{},
    observations:[{frame_index:3,timestamp_s:1.2,track_id:1,bbox_xyxy:[1,2,3,4]}]
  })});
  await assert.rejects(api.getRun('run-1'),error=>error.code==='INVALID_RESPONSE');
});

test('review summary keeps missing artifacts missing, partial status and track identity provisional', () => {
  const {reviewSummary,canonicalObservation}=require('./api-client.js');
  const row={schema_version:'touchline.observation.v1',run_id:'child',frame:{index:7,timestamp_s:2.5},bbox_xyxy:[1,2,20,30],local_track_id:1,player_id:null};
  assert.equal(canonicalObservation(row,'child'),true);
  const summary=reviewSummary({run_id:'child',manifest:{run_id:'child',parent_run_id:'parent',status:'partial'},observations:[row],video_url:'/runs/child/source_video.mp4'});
  assert.equal(summary.status,'partial');
  assert.equal(summary.parentRunId,'parent');
  assert.equal(summary.compatibleObservations,true);
  assert.equal(summary.sourceHash,null);
  assert.equal(summary.normalizedHash,null);
  assert.equal(summary.annotatedVideoAvailable,false);
  assert.equal(summary.correctionPersistenceAvailable,false);
  assert.equal(reviewSummary({annotated_video_url:'/runs/a/overlay24.mp4',annotated_video_verified:false}).annotatedVideoAvailable,false);
  assert.equal(reviewSummary({annotated_video_url:'/runs/a/overlay24.mp4',annotated_video_verified:true}).annotatedVideoAvailable,true);
  assert.equal(reviewSummary({annotated_video_url:'//untrusted.example/video',annotated_video_verified:true}).annotatedVideoAvailable,false);
});

test('original media time requires an exact frame-map row, never nominal FPS', () => {
  const {sourceTimeFromFrameMap}=require('./api-client.js');
  const rows=[{output_frame_index:24,source_relative_time:[1001,300]}];
  assert.equal(sourceTimeFromFrameMap(rows,24),1001/300);
  assert.equal(sourceTimeFromFrameMap(rows,25),null);
  assert.equal(sourceTimeFromFrameMap(null,24),null);
  assert.equal(sourceTimeFromFrameMap([{output_frame_index:24,source_relative_time:[1,0]}],24),null);
});

test('selected annotated playback time maps to exact source PTS before Meta frame review', () => {
  const {sourceTimeFromPlayback}=require('./api-client.js');
  const rows=[
    {output_frame_index:0,output_pts:0,output_time_base:[1,24],source_relative_time:[0,1]},
    {output_frame_index:1,output_pts:1,output_time_base:[1,24],source_relative_time:[1,25]},
    {output_frame_index:2,output_pts:2,output_time_base:[1,24],source_relative_time:[2,25]},
  ];
  assert.equal(sourceTimeFromPlayback(rows,1/24),1/25);
  assert.equal(sourceTimeFromPlayback(rows,2/24),2/25);
  assert.equal(sourceTimeFromPlayback(rows,1.8),null);
  assert.equal(sourceTimeFromPlayback([{output_frame_index:0,source_relative_time:[0,1]}],0),null);
});

test('production accepts canonical evidence but never invents annotated media or correction persistence', async () => {
  const {reviewSummary}=require('./api-client.js');
  const hash='a'.repeat(64);
  const payload={run_id:'run-1',manifest:{run_id:'run-1',status:'complete'},analytics:{},
    normalization:{source_sha256:hash,video_sha256:'b'.repeat(64)},
    frame_map:[{output_frame_index:2,source_relative_time:[3,2]}],
    observations:[{schema_version:'touchline.observation.v1',run_id:'run-1',frame:{index:2,timestamp_s:0.08},bbox_xyxy:[1,2,3,4],local_track_id:7,player_id:null}]};
  const api=createClient({mode:'production',routes:{getRun:'/runs/{id}'},getToken:async()=> 'token',fetchImpl:async()=>response(payload)});
  const summary=reviewSummary(await api.getRun('run-1'));
  assert.equal(summary.compatibleObservations,true);
  assert.equal(summary.sourceHash,hash);
  assert.equal(summary.normalizedHash,'b'.repeat(64));
  assert.equal(summary.frameMapAvailable,true);
  assert.equal(summary.annotatedVideoAvailable,false);
  assert.equal(summary.correctionPersistenceAvailable,false);
});

test('roster DTOs are normalized in the adapter, away from UI components', async () => {
  const api=createClient({
    mode:'production',routes:{getTeams:'/teams',getPlayers:'/teams/{teamId}/players'},getToken:async()=> 'token',
    fetchImpl:async url=>response(url==='/teams'
      ? {items:[{team_id:'team-1',team_name:'Academy'}]}
      : {items:[{player_id:'player-1',player_name:'Fixture athlete'}]})
  });
  assert.deepEqual((await api.getTeams()).map(x=>[x.id,x.name]),[['team-1','Academy']]);
  assert.deepEqual((await api.getPlayers('team-1')).map(x=>[x.id,x.name]),[['player-1','Fixture athlete']]);
});

test('provider or backend errors are reduced to safe typed UI errors', async () => {
  const api=createClient({mode:'production',routes:{getSessions:'/sessions'},getToken:async()=> 'token',fetchImpl:async()=>response({error:{code:'FORBIDDEN_SCOPE',message:'private path and header'}},403)});
  await assert.rejects(api.getSessions(),error=>error instanceof ApiError&&error.code==='FORBIDDEN_SCOPE'&&!error.message.includes('private path'));
});

test('stalled production reads time out even when transport ignores abort', async () => {
  const api=createClient({mode:'production',routes:{getSessions:'/sessions'},getToken:async()=> 'token',timeoutMs:8,fetchImpl:async()=>new Promise(()=>{})});
  await assert.rejects(api.getSessions(),error=>error.code==='TIMEOUT');
});

test('external cancellation settles a stalled request without a fixture fallback', async () => {
  const controller=new AbortController();
  const api=createClient({mode:'production',routes:{getSessions:'/sessions'},getToken:async()=> 'token',timeoutMs:1000,fetchImpl:async()=>new Promise(()=>{})});
  const read=api.getSessions({signal:controller.signal});
  controller.abort();
  await assert.rejects(read,error=>error.code==='CANCELLED');
});

test('job normalization rejects missing IDs and invented statuses', () => {
  assert.equal(job({job_id:'j',status:'succeeded'}).status,'completed');
  assert.equal(job({job_id:'j',status:'partial'}).status,'partial');
  assert.throws(()=>job({status:'completed'}),error=>error.code==='INVALID_RESPONSE');
  assert.throws(()=>job({job_id:'j',status:'99% done'}),error=>error.code==='INVALID_RESPONSE');
});

test('upload and job write routes are explicit and never infer model or local paths', async () => {
  const calls=[];
  const routes={createUpload:'/uploads',uploadRecording:'/uploads/{uploadId}',finalizeUpload:'/uploads/{uploadId}/finalize',createAnalysisJob:'/jobs'};
  const api=createClient({mode:'production',routes,getToken:async()=> 'token',fetchImpl:async(url,options)=>{
    calls.push({url,options});
    return response(url.endsWith('/jobs')?{job_id:'job',status:'queued',attempt_id:'attempt'}:{upload_id:'u'});
  }});
  await api.createUpload({session_id:'s',byte_count:2});
  await api.uploadRecording('u',new Blob(['ab'],{type:'video/mp4'}));
  await api.finalizeUpload('u',{sha256:'a'.repeat(64)});
  assert.equal((await api.createAnalysisJob({upload_id:'u',client_request_id:'request'})).status,'queued');
  assert.deepEqual(calls.map(x=>x.url),['/uploads','/uploads/u','/uploads/u/finalize','/jobs']);
  assert.equal(calls[1].options.method,'PUT');
  assert.equal(calls[1].options.body.size,2);
});
