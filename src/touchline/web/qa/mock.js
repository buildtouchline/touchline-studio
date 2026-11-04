// Local, deterministic UI fixture. It never calls a Touchline API or model.
(() => {
  const scenario = window.__qaScenario || 'normal';
  window.__touchlineFixture=true;
  const qa = window.__qa = {scenario, calls: [], transcriptSaves: 0};
  document.addEventListener('DOMContentLoaded',()=>{const mark=document.createElement('div');mark.textContent='SYNTHETIC QA FIXTURE';mark.setAttribute('aria-label','Synthetic QA fixture, not a real analysis');Object.assign(mark.style,{position:'fixed',right:'8px',bottom:'8px',zIndex:'999',background:'#201e24',color:'#f2e9e7',border:'1px solid #c37c73',padding:'5px 8px',font:'700 9px Consolas,monospace',letterSpacing:'.09em',pointerEvents:'none'});document.body.append(mark);});
  const run = (id, name = 'Training review / synthetic fixture') => ({id, name, has_video:true, duration_s:4, total_tracks:2,
    ...(scenario==='two_sessions'?{team_name:'Demo Academy',player_id:12,player_name:'Fixture player 12',session_label:id==='fixture-one'?'Session 1':'Session 2',status:'partial',synthetic:true}:{}),
    ...(scenario==='upload_complete'?{status:'completed',synthetic:true}:{})});
  const runs = scenario === 'empty' ? [] : scenario === 'large' ?
    Array.from({length:250}, (_, i) => run(`fixture-${i}`, `Session ${String(i).padStart(3,'0')}`)) :
    scenario === 'special' ? [run('fixture-one', 'Set piece [A] + 100%')] :
    scenario === 'long_name' ? [run('fixture-one', 'A very long football session name '.repeat(9))] :
    scenario === 'missing_metadata' ? [{id:'fixture-one'}] :
    (scenario === 'switch_delayed' || scenario === 'two_sessions') ? [run('fixture-one'),run('fixture-two','Second review / synthetic fixture')] : [run('fixture-one')];
  const observations = [
    {frame:{timestamp_s:0.5,width:640,height:360},local_track_id:7,bbox_xyxy:[96,80,155,285]},
    {frame:{timestamp_s:0.5,width:640,height:360},local_track_id:8,bbox_xyxy:[380,100,433,288]},
    {frame:{timestamp_s:1.5,width:640,height:360},local_track_id:7,bbox_xyxy:[106,80,165,285]},
    {frame:{timestamp_s:2.5,width:640,height:360},local_track_id:7,bbox_xyxy:[116,80,175,285]}
  ];
  const roster = [
    {local_track_id:7,confirmed_player_id:null,player_name:null,jersey_number:null,observation_count:3,trajectory:[]},
    {local_track_id:8,confirmed_player_id:12,player_name:scenario==='two_sessions'?'Fixture player 12':'A. Coach',jersey_number:9,reviewed_by:'fixture-coach',association_revision:1,observation_count:1,trajectory:[]}
  ];
  const transcript = {schema_version:'touchline.transcript.v1',segments:[{cue_id:'cue-1',text:'Move into space.',video_start_s:0.5,video_end_s:1.7,language_code:'en-IN'}],manual_cues:[]};
  const detail = {
    manifest:{status:scenario==='partial'||scenario==='two_sessions'?'partial':'complete',stop_reason:'frame_limit',selected_frames:4,observation_count:4,source:{name:'Synthetic QA fixture'}},
    video_url:'/web/qa/fixture.mp4',source_geometry:[640,360],observations,
    analytics:{session:{duration_s:4,observation_count:4},player_roster:roster,synthetic:true},
    calibration:{status:'uncalibrated',calibrated:false},transcript
  };
  if(scenario==='missing_video')delete detail.video_url;
  if(scenario==='invalid_video')detail.video_url='/web/qa/missing-video.mp4';
  if(scenario==='missing_geometry'){delete detail.source_geometry;for(const o of detail.observations)delete o.frame.width,delete o.frame.height;}
  if(scenario==='manifest_geometry'){delete detail.source_geometry;detail.manifest.image_shape_hw=[360,640];}
  if(scenario==='geometry_mismatch')detail.source_geometry=[1080,1920];
  if(scenario==='no_observations')detail.observations=[];
  if(scenario==='malformed_observations')detail.observations=[{}, {frame:{timestamp_s:-2},bbox_xyxy:[1,2]},...observations];
  if(scenario==='out_of_order')detail.observations=[...observations].reverse();
  if(scenario==='duplicate_observations')detail.observations=[...observations,observations[0]];
  if(scenario==='many_observations'){detail.observations=Array.from({length:1200},(_,i)=>({...observations[i%4],frame:{...observations[i%4].frame,timestamp_s:i/300}}));detail.manifest.observation_count=14200;}
  if(scenario==='verified')detail.calibration={status:'verified',calibrated:true,independent_check_ref:'qa-fixture-reference-only'};
  if(scenario==='preliminary')detail.calibration={status:'preliminary_unverified',calibrated:true};
  if(scenario==='malformed_calibration')detail.calibration={status:'verified',calibrated:'yes'};
  if(scenario==='missing_calibration')delete detail.calibration;
  if(scenario==='no_transcript')detail.transcript={segments:[],manual_cues:[]};
  if(scenario==='missing_transcript')delete detail.transcript;
  if(scenario==='malformed_transcript')detail.transcript={segments:'invalid',manual_cues:null};
  if(scenario==='duplicate_cues')detail.transcript.segments.push({...detail.transcript.segments[0]});
  if(scenario==='out_of_order_cues')detail.transcript.segments.push({cue_id:'early',text:'Earlier note',video_start_s:0.1,video_end_s:0.3});
  if(scenario==='long_transcript')detail.transcript.segments=Array.from({length:120},(_,i)=>({cue_id:`cue-${i}`,text:`Coach note ${i}`,video_start_s:i/40,video_end_s:i/40+0.2}));
  if(scenario==='missing_track_identity')detail.analytics.player_roster=[{local_track_id:99,observation_count:0}];
  if(scenario==='confirmed_player')detail.analytics.player_roster=[roster[1]];
  if(scenario==='unreviewed_player')detail.analytics.player_roster=[{...roster[1],reviewed_by:null,association_revision:null}];
  if(scenario==='missing_player_name')detail.analytics.player_roster=[{...roster[1],player_name:null,jersey_number:null}];
  if(scenario==='many_tracks')detail.analytics.player_roster=Array.from({length:40},(_,i)=>({local_track_id:i,confirmed_player_id:null,observation_count:2}));
  const json=(body,status=200)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
  window.XMLHttpRequest=class FixtureUploadRequest {
    upload={addEventListener:(name,listener)=>{if(name==='progress')this.progress=listener;}};
    open(method,url){this.method=method;this.url=url;}
    setRequestHeader(){}
    send(file){setTimeout(()=>{
      if(scenario==='upload_http_error'){this.status=413;this.responseText=JSON.stringify({error:'Fixture upload rejected'});this.onload?.();return;}
      this.progress?.({lengthComputable:true,loaded:file.size,total:file.size});
      this.status=202;this.responseText=JSON.stringify({status:'processing',job_id:'fixture-one'});this.onload?.();
    },120);}
  };
  window.fetch=async (input, options={})=>{
    const url=String(input), method=options.method||'GET';qa.calls.push({url,method});
    if(scenario==='network_unavailable')throw new TypeError('Failed to fetch');
    if(scenario==='slow_network')await new Promise(r=>setTimeout(r,1800));
    if(scenario==='timeout')return new Promise(()=>{});
    if(url==='/api/runs'){
      if(scenario==='invalid_runs')return json({runs});
      if(scenario==='malformed_json')return new Response('{broken',{status:200,headers:{'Content-Type':'application/json'}});
      if(scenario.startsWith('http_'))return json({error:'Fixture HTTP error'},Number(scenario.slice(5)));
      return json(runs);
    }
    if(url.endsWith('/transcript')&&method==='POST'){
      if(scenario==='save_failure')return json({error:'Fixture save failed'},500);
      qa.transcriptSaves++;detail.transcript=JSON.parse(options.body);return json({status:'ok',transcript:detail.transcript});
    }
    if(url.includes('/subtitles.')){
      if(scenario==='export_failure')return json({error:'Fixture export failed'},500);
      return new Response('1\n00:00:00,500 --> 00:00:01,700\nMove into space.\n',{status:200,headers:{'Content-Type':'text/plain'}});
    }
    if(url.startsWith('/api/run/')){
      if((scenario==='detail_timeout'||scenario==='switch_delayed')&&url.endsWith('/fixture-one'))return new Promise(()=>{});
      if((scenario==='switch_delayed'||scenario==='two_sessions')&&url.endsWith('/fixture-two'))return json({...detail,manifest:{...detail.manifest,source:{name:'Second review / synthetic fixture'}}});
      return json(detail);
    }
    if(url.startsWith('/api/audio/transcribe')){
      if(scenario==='audio_failure')return json({error:'Fixture transcription failed'},500);
      if(scenario==='audio_malformed')return json({status:'ok',transcript:null});
      return json({status:'ok',transcript:detail.transcript});
    }
    if(url.startsWith('/api/analyze'))return json({status:'processing',job_id:'fixture-one'},202);
    if(url.startsWith('/api/job/'))return json({status:scenario==='upload_failed'?'failed':scenario==='upload_processing'?'processing':scenario==='upload_complete'?'completed':'partial',job_id:'fixture-one'});
    return json({error:'Unmocked fixture route'},404);
  };
  window.addEventListener('error',e=>parent.postMessage({type:'qa-error',message:e.message},'*'));
  window.addEventListener('unhandledrejection',e=>parent.postMessage({type:'qa-error',message:String(e.reason)},'*'));
  const originalError=console.error.bind(console);
  console.error=(...args)=>{parent.postMessage({type:'qa-error',message:args.map(String).join(' ')},'*');originalError(...args);};
  parent.postMessage({type:'qa-fixture',scenario},'*');
})();
