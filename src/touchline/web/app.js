'use strict';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries([
  'navOverview','navTeams','navLibrary','navWorkspace','navDevelopment','navReports','overviewView','teamsView','libraryView','analysisView','workspaceView','developmentView','reportsView',
  'analysisCount','analysisState','analysisRunList','btnAnalysisRefresh','btnAnalysisSessions','btnAnalysisDashboard',
  'teamsContent','developmentOverview','reportsContent','demoRibbon','btnDemoStart','btnResetDemo',
  'jobBanner','jobTitle','jobMessage','jobProgress','btnJobOpen','btnJobRetry','btnJobDismiss',
  'btnOverviewSessions','overviewSessionCount','overviewVideoCount','overviewDraftStatus','overviewSessionList','libraryCount','libraryList',
  'libraryState','librarySearch','btnBackLibrary','frameReadout',
  'sessionSelect','btnRetrySessions','calBadge','sessionTitle','sessionMeta','runStatus','sessionAlert','metaModelSelect','metaObjective','metaObjectiveLabel','btnMetaFrame','metaActivityReadiness','metaActivityList',
  'matchVideo','overlayCanvas','videoStatus','videoSubtitlesOverlay','subtitleText','videoSlider','fixtureVideoPlaceholder','btnOpenFixtureVideo','transportDeck',
  'timelineMarkers','timeDisplay','btnPlayPause','btnStepBack','btnStepFwd','chkBoxes','chkTags',
  'chkTrails','chkSubsOverlay','chkLoop','speedSelect','headerLangSelect','exportDropdown',
  'statObservations','statTracks','statConfirmed','statPhysical','coverageNotice','recordingProvenance','recordingArtifacts','recordingReviewAction','plannedCapabilities','subtitlesList',
  'playersList','selectedTrackInfo','fixtureInspectorPlaceholder','subsCount','playerCount','micInputSelect','btnRefreshDevices',
  'btnRecordMic','recordElapsed','micStatusLabel','audioReview','audioPreview','audioFileInput','btnUploadAudio',
  'btnDiscardAudio','btnAddManualSubtitle','subtitleModal','modalSubTitle','subStartInput',
  'subEndInput','subTextInput','subLangSelect','saveStatus','btnSaveSubModal','btnCancelSubModal',
  'btnSyncTime','uploadModal','uploadForm','uploadFileInput','uploadFramesSelect','uploadStatusText',
  'btnSubmitUpload','btnUploadModal','btnCancelUpload','toastHost',
  'btnAddMarker','markerModal','markerTime','markerLabel','markerStatus','btnCancelMarker','btnSaveMarker',
  'timelineList','timelineCount','developmentPlayer','developmentFocus','developmentCheck',
  'btnCreateCorrection','developmentDraft','btnAddFollowup','btnDiscardCorrection',
  'developmentOutcome','developmentAssessment','developmentComparison','btnSaveAssessment','btnExportCorrection','developmentStatus'
].map(id => [id, $(id)]));
const ctx = ui.overlayCanvas.getContext('2d');
const state = { id:null, data:null, runs:[], listError:null, frameTimes:[], frames:new Map(), selected:null, loadToken:0, listToken:0,
  recorder:null, stream:null, recording:null, recorded:null, audioUrl:null, clock:null, uploadingAudio:false, disposing:false,
  saving:false, editing:null, lastRenderedFrame:null, geometryWarning:false, calibrationVerified:false, modalReturnFocus:null,
  localMarkers:[],timelineEntries:[],developmentDraft:null,view:'library',demo:window.__touchlineFixture===true,fixtureVideoOpen:false,mediaKind:'source',job:null,focusPlayerId:null,pollToken:0,listLoading:false,teams:null,playersByTeam:new Map(),rosterError:null,rosterLoading:false,metaActivities:[],metaPollToken:0,metaConfigured:false,metaEnabled:false,metaRemaining:0,metaError:null };
let api;
try { api=TouchlineApi.createClient({...(window.__touchlineApiConfig||{}),mode:state.demo?'demo':window.__touchlineApiConfig?.mode||'local',syntheticFixture:state.demo}); }
catch(error) {
  // A misconfigured production client must show an error, never silently use Studio or fixtures.
  api={mode:'production',getSessions:()=>Promise.reject(error),getRun:()=>Promise.reject(error),getJob:()=>Promise.reject(error),readiness:()=>Promise.reject(error)};
}
const draftStorage=api.mode==='production'?{
  getItem:key=>localStorage.getItem(`production.${key}`),
  setItem:(key,value)=>localStorage.setItem(`production.${key}`,value),
  removeItem:key=>localStorage.removeItem(`production.${key}`)
}:localStorage;
try { state.developmentDraft=TouchlineDevelopmentStorage.read(draftStorage); }
catch { state.developmentDraft=null; }
function persistDevelopment() {
  try {
    if(state.developmentDraft)TouchlineDevelopmentStorage.save(draftStorage,state.developmentDraft);
    else TouchlineDevelopmentStorage.remove(draftStorage);
    return true;
  } catch { ui.developmentStatus.textContent='Browser storage unavailable. Draft remains in this tab only; export it before closing.'; return false; }
}

function toast(message, kind='info') {
  const item=document.createElement('div'); item.className=`toast-pill ${kind}`;
  item.textContent=message; ui.toastHost.append(item); setTimeout(()=>item.remove(),4500);
}
function showModal(modal,focusTarget){state.modalReturnFocus=document.activeElement;modal.style.display='flex';focusTarget.focus();}
function hideModal(modal){modal.style.display='none';state.modalReturnFocus?.focus();state.modalReturnFocus=null;}
document.addEventListener('keydown',event=>{
  const modal=[ui.uploadModal,ui.subtitleModal,ui.markerModal].find(el=>el.style.display==='flex');
  if(!modal)return;
  if(event.key==='Escape'){
    event.preventDefault();
    if(modal===ui.subtitleModal)closeCue();
    else if(modal===ui.markerModal)hideModal(modal);
    else if(!ui.btnSubmitUpload.disabled)hideModal(modal);
    return;
  }
  if(event.key!=='Tab')return;
  const focusable=[...modal.querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled)')];
  if(!focusable.length)return;
  const first=focusable[0],last=focusable.at(-1);
  if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
  else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
});
function notice(el, message) { el.textContent=message; el.hidden=!message; }
function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const s=Math.max(0,seconds); return `${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
}
function finite(value) { return typeof value==='number' && Number.isFinite(value); }
function safeArray(value) { return Array.isArray(value) ? value : []; }
function calibrationPayload(data) {
  if(data?.calibration?.status)return data.calibration;
  if(data?.analytics?.calibration?.status)return data.analytics.calibration;
  if(data?.status==='verified'||data?.status==='preliminary_unverified'||data?.status==='uncalibrated')return data;
  return data?.calibration||data?.analytics?.calibration||{};
}
function independentlyVerified(cal,data) {
  return data?.analytics?.synthetic!==true && cal.status==='verified'&&cal.calibrated===true&&
    typeof cal.independent_check_ref==='string'&&cal.independent_check_ref.trim().length>0&&
    cal.validation?.held_out_check_passed===true&&cal.validation?.geometry_match===true&&
    cal.validation?.independent_reference_verified===true;
}
function reviewedPlayer(p){return TouchlineDevelopment.confirmedPlayer([p],p?.confirmed_player_id);}
function runUrl(id) { return `/api/run/${encodeURIComponent(id)}`; }
function samBoxes(text){
  const boxes=[];const pattern=/<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)/g;
  for(const match of String(text||'').matchAll(pattern)){
    const [x1,y1,x2,y2,width,height]=match.slice(1).map(Number);
    if(width>0&&height>0&&x1>=0&&y1>=0&&x2>x1&&y2>y1&&x2<=width&&y2<=height)boxes.push({x1,y1,x2,y2,width,height});
    if(boxes.length>=100)break;
  }
  return boxes;
}
function renderMetaActivity(){
  const realRun=api.mode==='local'&&!state.demo&&state.id&&state.data?.video_url;
  const sourceAlignmentAvailable=state.mediaKind==='source'||Array.isArray(state.data?.frame_map)&&state.data.frame_map.length>1;
  const attempts=state.metaActivities.filter(item=>item.model_kind===ui.metaModelSelect.value);
  const modelUsed=attempts.length>=2||attempts.some(item=>['submitted','processing','response_received'].includes(item.status));
  const objectiveReady=ui.metaModelSelect.value!=='wonder'||ui.metaObjective.value.trim().length>=3;
  const ready=realRun&&sourceAlignmentAvailable&&state.metaConfigured&&state.metaEnabled&&state.metaRemaining>0&&!modelUsed&&objectiveReady;
  ui.btnMetaFrame.disabled=!ready;
  ui.metaActivityReadiness.textContent=state.metaError||(!realRun?'Open a real recording; synthetic fixtures cannot call Meta.':
    !sourceAlignmentAvailable?'Original frame time is unavailable; switch to source video or provide a verified frame map.':
    !state.metaConfigured?'Server-side Meta credential unavailable.':
    !state.metaEnabled||state.metaRemaining<=0?'Server-side Meta call allowance is closed.':
    modelUsed?'This model has a result, is processing, or reached its two-attempt limit.':
    !objectiveReady?'Enter a specific coach objective for Muse Wonder.':'Ready for one bounded selected-frame request.');
  ui.metaActivityList.replaceChildren();
  for(const activity of state.metaActivities){
    const card=document.createElement('article');card.className='meta-activity-card';
    const friendly={sam:'SAM 3.1',spark:'Muse Spark',wonder:'Muse Wonder',voice:'Muse Voice Transcribe'}[activity.model_kind]||'Meta model';
    const title=document.createElement('h3');title.textContent=`${friendly} · ${activity.model||'model ID unavailable'} · ${String(activity.status||'unknown').replaceAll('_',' ')}`;
    const details=document.createElement('p');details.textContent=`${activity.provider||'Meta Model API'} · ${activity.input_type||'source frame'} · ${Number.isFinite(activity.frame_time_s)?formatTime(activity.frame_time_s):formatTime(activity.requested_time_s)} · ${activity.latency_ms==null?'pending':`${activity.latency_ms} ms`}`;
    const reference=activity.model_kind==='voice'?`Provider session ${activity.provider_session_id||activity.request_id||'not returned'}`:`Request ${activity.request_id||'not returned'}`;
    const ids=document.createElement('small');ids.textContent=`Session ${activity.run_id||'—'} · ${reference} · ${activity.runtime||'server'} · ${activity.submitted_at||'—'} → ${activity.responded_at||'pending'}${activity.response_model?` · Returned model ${activity.response_model}`:''}`;
    card.append(title,details,ids);
    const lifecycle=document.createElement('p');lifecycle.className='meta-activity-lifecycle';
    lifecycle.textContent=activity.status==='response_received'?
      `Provider ${activity.response_status||'response received'} · usable output parsed and displayed · ${activity.review?`coach ${activity.review.decision.replaceAll('_',' ')}`:'AI-generated proposal awaiting coach review'}`:
      activity.status==='failed'?'Request failed; no reviewable model output was accepted.':
      activity.status==='processing'?'Provider processing; no result has been accepted yet.':'Submitted; awaiting provider processing.';
    card.append(lifecycle);
    if(activity.model_kind==='voice'){
      const origin=document.createElement('p');origin.className='meta-activity-origin';
      origin.textContent=activity.input_origin==='locally_generated_test_audio'?'Input: locally generated spoken-audio test, not a coach recording.':
        'Input: coach audio source has not been independently verified.';card.append(origin);
    }
    if(activity.output_preview){
      const boxes=activity.model==='sam-3.1'?samBoxes(activity.output_preview):[];
      if(activity.frame_sha256){
        const frame=document.createElement('div');frame.className='meta-frame-view';
        const image=document.createElement('img');image.alt='Source video frame supplied to Meta; model geometry is overlaid when available';
        image.addEventListener('error',()=>{const unavailable=document.createElement('p');unavailable.className='meta-frame-unavailable';
          unavailable.textContent='Saved source-frame preview unavailable; its hash could not be verified in this runtime. Model text remains available for coach review.';
          frame.replaceChildren(unavailable);});
        image.src=`${runUrl(activity.run_id)}/meta-activity/${encodeURIComponent(activity.activity_id)}/frame.jpg`;
        frame.append(image);
        for(const [index,box] of boxes.entries()){
          const marker=document.createElement('span');marker.className='meta-frame-box';marker.style.left=`${100*box.x1/box.width}%`;marker.style.top=`${100*box.y1/box.height}%`;marker.style.width=`${100*(box.x2-box.x1)/box.width}%`;marker.style.height=`${100*(box.y2-box.y1)/box.height}%`;marker.title=`SAM proposal ${index+1}; unconfirmed identity`;frame.append(marker);
        }
        card.append(frame);
      }
      if(activity.model==='sam-3.1'){
        const summary=document.createElement('p');summary.textContent=`${boxes.length} visible person-box proposals from the returned segmentation text. Mask payloads are retained in the technical response; these are not confirmed players.`;card.append(summary);
        const raw=document.createElement('details');const label=document.createElement('summary');label.textContent='Technical Meta segmentation response';
        const output=document.createElement('pre');output.className='meta-activity-output';output.textContent=activity.output_preview;raw.append(label,output);card.append(raw);
      }else if(activity.model_kind==='spark'){
        let structured=null;try{structured=JSON.parse(activity.output_preview);}catch{}
        if(structured&&typeof structured==='object'&&!Array.isArray(structured)){
          const list=document.createElement('dl');list.className='meta-structured-output';
          for(const [key,label] of [['visible_observation','Visible observation'],['supporting_visual_evidence','Supporting visual evidence'],
                                    ['uncertainty','Uncertainty'],['suggested_coaching_question','Suggested coaching question']]){
            const term=document.createElement('dt');term.textContent=label;const value=document.createElement('dd');
            value.textContent=typeof structured[key]==='string'&&structured[key].trim()?structured[key]:'Not supplied';list.append(term,value);
          }
          card.append(list);
        }else{const output=document.createElement('p');output.className='meta-activity-text';output.textContent=activity.output_preview;card.append(output);}
      }else{const output=document.createElement('p');output.className='meta-activity-text';output.textContent=activity.output_preview;card.append(output);}
    }
    if(activity.coach_objective){const goal=document.createElement('p');goal.textContent=`Coach objective: ${activity.coach_objective}`;card.append(goal);}
    if(Array.isArray(activity.turns)&&activity.turns.length){const turns=document.createElement('ol');for(const turn of activity.turns){const line=document.createElement('li');line.textContent=`${formatTime(turn.startMs/1000)}–${formatTime(turn.endMs/1000)}${turn.speaker?` · Speaker ${turn.speaker} (unverified)`:''}: ${turn.transcript}`;turns.append(line);}card.append(turns);}
    if(activity.error){const error=document.createElement('p');error.className='meta-activity-error';error.textContent=`${activity.error} ${activity.error_code||''}${activity.provider_http_status?` · HTTP ${activity.provider_http_status}`:''}${activity.provider_error_detail?` · ${activity.provider_error_detail}`:''}`;card.append(error);}
    if(activity.status==='response_received'){
      const review=document.createElement('div');review.className='meta-activity-review';
      if(activity.review){review.textContent=`Coach review: ${activity.review.decision.replaceAll('_',' ')} · ${activity.review.reviewer==='local_unauthenticated_coach'?'local Studio review':activity.review.reviewer}`;}
      else {
        let edit=null;
        if(activity.model_kind!=='sam'){
          const label=document.createElement('label');label.textContent=activity.model_kind==='voice'?'Edit transcript before approval':'Edit proposed observation before approval';
          edit=document.createElement('textarea');edit.maxLength=12000;edit.rows=3;edit.value=activity.output_preview||'';label.append(edit);review.append(label);
        }
        for(const [label,decision] of [[activity.model_kind==='voice'?'Approve transcript':'Confirm observation','confirmed_observation'],
                                        [activity.model_kind==='voice'?'Reject transcript':'Reject observation','rejected_observation']]){
          const button=document.createElement('button');button.type='button';button.className='btn-ctrl';button.textContent=label;
          button.addEventListener('click',()=>reviewMetaActivity(activity.activity_id,decision,
            decision==='confirmed_observation'&&edit&&edit.value!==activity.output_preview?edit.value:null));review.append(button);
        }
      }
      card.append(review);
    }
    ui.metaActivityList.append(card);
  }
  if(!state.metaActivities.length){const empty=document.createElement('p');empty.className='muted';empty.textContent='No real Meta model request has been made for this session.';ui.metaActivityList.append(empty);}
}
async function refreshMetaActivity(){
  const runId=state.id,token=++state.metaPollToken;
  if(api.mode!=='local'||state.demo||!runId||!state.data){state.metaActivities=[];state.metaConfigured=false;state.metaEnabled=false;state.metaRemaining=0;renderMetaActivity();return;}
  try{
    const response=await fetch(`${runUrl(runId)}/meta-activity`,{headers:{Accept:'application/json'}});
    const payload=await jsonResponse(response);
    if(token!==state.metaPollToken||runId!==state.id)return;
    state.metaActivities=Array.isArray(payload.activities)?payload.activities:[];
    state.metaConfigured=payload.configured===true;state.metaEnabled=payload.enabled===true;
    state.metaRemaining=Number.isInteger(payload.remaining_calls)?payload.remaining_calls:0;
  }catch{if(token!==state.metaPollToken)return;state.metaConfigured=false;state.metaEnabled=false;}
  renderMetaActivity();
  if(state.metaActivities.some(item=>['submitted','processing'].includes(item.status)))setTimeout(()=>{if(token===state.metaPollToken)refreshMetaActivity();},1200);
}
async function reviewMetaActivity(activityId,decision,editedText=null){
  if(!state.id||api.mode!=='local'||state.demo)return;
  try{await jsonResponse(await fetch(`${runUrl(state.id)}/meta-activity/${encodeURIComponent(activityId)}/review`,{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision,...(editedText!==null?{edited_text:editedText}: {})})}));
    await refreshMetaActivity();
    if(decision==='confirmed_observation'&&state.metaActivities.some(item=>item.activity_id===activityId&&item.model_kind==='voice')){
      const data=await jsonResponse(await fetch(`${runUrl(state.id)}`));
      if(state.data&&state.id===data.run_id){state.data.transcript=data.transcript;renderCues();renderTimeline();}
    }}
  catch(error){toast(`Review was not saved: ${error.message}`,'error');}
}
ui.btnMetaFrame.addEventListener('click',async()=>{
  if(!state.id||!state.data?.video_url||api.mode!=='local'||state.demo)return;
  const sourceTime=state.mediaKind==='annotated'?TouchlineApi.sourceTimeFromPlayback(state.data.frame_map,ui.matchVideo.currentTime||0):ui.matchVideo.currentTime||0;
  if(!Number.isFinite(sourceTime)){state.metaError='Original source time is unavailable for this annotated frame.';renderMetaActivity();return;}
  ui.btnMetaFrame.disabled=true;state.metaError='Submitting selected source frame to Meta…';renderMetaActivity();
  try{
    const response=await fetch(`${runUrl(state.id)}/meta-frame`,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:ui.metaModelSelect.value,timestamp_s:sourceTime,
        ...(ui.metaModelSelect.value==='wonder'?{objective:ui.metaObjective.value.trim()}: {})})});
    await jsonResponse(response);state.metaError=null;await refreshMetaActivity();
  }catch(error){state.metaError=`Request not started: ${error.message}`;}
  finally{renderMetaActivity();}
});
ui.metaModelSelect.addEventListener('change',()=>{const wonder=ui.metaModelSelect.value==='wonder';ui.metaObjective.hidden=!wonder;ui.metaObjectiveLabel.hidden=!wonder;state.metaError=null;renderMetaActivity();});
ui.metaObjective.addEventListener('input',renderMetaActivity);
function openRunOrJob(run){
  const status=String(run.status||'').toLowerCase();
  if(['queued','processing','failed','cancelled','unavailable'].includes(status)){
    state.job={id:run.id,status};renderJob();showView('analysis');return;
  }
  ui.sessionSelect.value=run.id;loadSession(run.id);
}
async function jsonResponse(response) {
  let body; try { body=await response.json(); } catch { throw new Error(`Server returned HTTP ${response.status} without JSON`); }
  if (!response.ok || body?.error) throw new Error(body?.error || `HTTP ${response.status}`);
  return body;
}
const listReads=TouchlineRequest.createGate((_,options)=>api.getSessions(options),value=>value);
const detailReads=TouchlineRequest.createGate((id,options)=>api.getRun(id,options),value=>value);
function setOption(select,value,label) { const opt=document.createElement('option'); opt.value=value; opt.textContent=label; select.append(opt); }
function setBusy(isBusy) { ui.sessionSelect.disabled=isBusy; ui.btnRetrySessions.disabled=isBusy; }
function showView(view) {
  const active=view==='workspace'&&!state.data?'analysis':view;
  state.view=active;
  for(const [name,viewId,navId] of [['overview','overviewView','navOverview'],['teams','teamsView','navTeams'],['library','libraryView','navLibrary'],['analysis','analysisView','navWorkspace'],['development','developmentView','navDevelopment'],['reports','reportsView','navReports']]){
    const visible=active===name, selected=visible||name==='analysis'&&active==='workspace';
    ui[viewId].hidden=!visible;
    ui[navId].classList.toggle('active',selected);
    ui[navId].setAttribute('aria-current',selected?'page':'false');
  }
  ui.workspaceView.hidden=active!=='workspace';
  if(active==='overview')renderOverview();
  if(active==='analysis')renderAnalysis();
  if(['teams','development','reports'].includes(active))renderProductViews();
  if(active==='workspace')requestAnimationFrame(renderFrame);
}
function runLabel(run) {
  if(run.id==='calibrated-snmot097')return 'SNMOT-097 / legacy demo';
  return run.name || run.id;
}
function renderOverview() {
  ui.overviewSessionCount.textContent=state.listError?'Unavailable':String(state.runs.length);
  ui.overviewVideoCount.textContent=state.listError?'Unavailable':String(state.runs.filter(run=>run.has_video===true).length);
  ui.overviewDraftStatus.textContent=state.developmentDraft?'1 browser-local draft':'No local draft';
  ui.overviewSessionList.replaceChildren();
  if(state.listError){ui.overviewSessionList.textContent='Session list unavailable. Return to Sessions and retry.';return;}
  if(!state.runs.length){ui.overviewSessionList.textContent=api.mode==='production'?'No authorized sessions were returned. Create a session when the backend workflow is available.':'No local runs are available. Full-recording upload awaits the authenticated backend job API.';return;}
  for(const run of state.runs.slice(0,5)){
    const row=document.createElement('button');row.type='button';row.className='overview-run';
    const name=document.createElement('strong');name.textContent=runLabel(run);
    const meta=document.createElement('span');meta.textContent=`${run.has_video?'Source ready':'No source'} · ${run.status?String(run.status).toUpperCase():'analysis status available after opening'}${run.synthetic?' · SYNTHETIC FIXTURE':''}`;
    row.append(name,meta);row.addEventListener('click',()=>openRunOrJob(run));ui.overviewSessionList.append(row);
  }
}
function renderLibrary() {
  renderOverview();
  if(state.listError){ui.libraryCount.textContent='— SESSIONS';ui.libraryList.replaceChildren();ui.libraryState.hidden=false;ui.libraryState.textContent=state.listError;return;}
  const query=ui.librarySearch.value.trim().toLowerCase();
  const runs=state.runs.filter(run=>`${runLabel(run)} ${run.id}`.toLowerCase().includes(query));
  ui.libraryList.replaceChildren();ui.libraryCount.textContent=`${state.runs.length} ${state.runs.length===1?'SESSION':'SESSIONS'}`;
  for(const [index,run] of runs.entries()) {
    const row=document.createElement('button');row.type='button';row.className='library-row';
    row.setAttribute('aria-label',`Open ${runLabel(run)}`);
    const name=document.createElement('span');name.className='library-row-name';
    const title=document.createElement('strong');title.textContent=runLabel(run);
    const id=document.createElement('small');id.textContent=`${String(index+1).padStart(2,'0')} / ${run.id}${run.status?` · ${String(run.status).toUpperCase()}`:''}${run.synthetic?' · FIXTURE':''}`;
    name.append(title,id);
    const video=document.createElement('span');video.className=`library-cell${run.has_video?' ready':''}`;video.textContent=run.has_video?'Source ready':'No source';
    const duration=document.createElement('span');duration.className='library-cell';duration.textContent=finite(run.duration_s)&&run.duration_s>0?formatTime(run.duration_s):'—';
    const tracks=document.createElement('span');tracks.className='library-cell tracks';tracks.textContent=Number.isInteger(run.total_tracks)?String(run.total_tracks):'—';
    const arrow=document.createElement('span');arrow.className='library-arrow';arrow.textContent='↗';arrow.setAttribute('aria-hidden','true');
    row.append(name,video,duration,tracks,arrow);
    row.addEventListener('click',()=>openRunOrJob(run));ui.libraryList.append(row);
  }
  if(!runs.length) {
    ui.libraryState.hidden=false;
    ui.libraryState.textContent=state.runs.length?'No sessions match this search.':api.mode==='production'?'No authorized sessions were returned.':'No local runs are available. Full-recording upload is not deployed.';
  } else ui.libraryState.hidden=true;
}
function renderAnalysis(){
  ui.analysisRunList.replaceChildren();
  ui.analysisCount.textContent=`${state.runs.length} ${state.runs.length===1?'RUN':'RUNS'}`;
  if(state.listLoading){ui.analysisState.textContent='Loading available analysis runs…';return;}
  if(state.listError){
    ui.analysisState.textContent=`${state.listError} ${api.mode==='production'?'Check the backend route, sign-in state and access to this school.':'This server may not provide the Studio API. The QA demo opens from its root URL; real runs require the Touchline Studio server.'}`;
  }else if(!state.runs.length){
    ui.analysisState.textContent=state.job?.id?'No finished run is listed yet. Follow the job status above or refresh runs.':api.mode==='production'?'No authorized analysis results were returned.':'No analysis results are available. The authenticated full-recording job API is not deployed.';
  }else ui.analysisState.textContent=api.mode==='production'?'Review each backend job and result status before drawing conclusions.':'Finished-run status comes from its manifest; pending and failed uploads come from the Studio job record.';
  const job=state.job;
  if(job&&(!state.runs.some(run=>run.id===job.id)||['uploading','queued','processing','failed','cancelled','unavailable'].includes(job.status))){
    const status=String(job.status||'processing').toUpperCase();
    const description=job.status==='processing'?'The upload was accepted. No completed analysis is claimed while the worker runs.':job.status==='queued'?'The job is queued; no analysis results are available yet.':job.status==='failed'?'Processing failed. The source may be retained; check before uploading again.':job.status==='cancelled'?'The job was cancelled; any retained source must be checked before retrying.':job.status==='uploading'?'Video bytes are still uploading.':'Status is unavailable. Check this job and the session list before retrying.';
    ui.analysisRunList.append(panel(`Job ${job.id||'upload'} · ${status}`,`${description}${state.demo?' SYNTHETIC SIMULATION: the selected file was not analysed.':''}`,job.status==='failed'?[['New upload',showUploadModal]]:job.id?[['Check job status',()=>pollJob(job.id)]]:[]));
  }
  for(const run of state.runs){
    const rawStatus=String(run.status||'unknown').toUpperCase();
    const status=rawStatus==='COMPLETED'?'COMPLETE':rawStatus;
    const details={COMPLETE:'Analysis finished; review the manifest and available evidence.',PARTIAL:'Incomplete coverage; inspect sampled frames and the stop reason.',QUEUED:'Queued for processing; no analysis result yet.',PROCESSING:'Processing continues; evidence may not be ready.',FAILED:'Processing failed; any saved artifacts need careful review.',CANCELLED:'Processing was cancelled; check retained artifacts before retrying.',UNKNOWN:'Status is not present in the session list; inspect the manifest before drawing conclusions.'};
    const actions=['QUEUED','PROCESSING','FAILED','CANCELLED','UNAVAILABLE'].includes(status)
      ?[['Check job status',()=>{state.job={id:run.id,status:status.toLowerCase()};renderJob();pollJob(run.id);}]]
      :[['Open available evidence',()=>{ui.sessionSelect.value=run.id;loadSession(run.id);}]];
    ui.analysisRunList.append(panel(`${runLabel(run)} · ${status}`,`Run ${run.id} · ${run.has_video?'source video available':'no source video listed'}${run.synthetic?' · SYNTHETIC FIXTURE':''}. ${details[status]||'Review the run manifest for its status.'}`,actions));
  }
}
function panel(title,description,actions=[]) {
  const article=document.createElement('article');article.className='product-panel';
  const heading=document.createElement('h2');heading.textContent=title;
  const body=document.createElement('p');body.textContent=description;
  article.append(heading,body);
  for(const [label,callback] of actions){const button=document.createElement('button');button.type='button';button.className='btn-ctrl';button.textContent=label;button.addEventListener('click',callback);article.append(button);}
  return article;
}
async function openEvidence(moment) {
  if(!moment)return;
  await loadSession(moment.run_id);
  if(state.id===moment.run_id&&state.data){
    if(ui.matchVideo.readyState>=1)seekObservation(moment.video_time_s);
    else ui.matchVideo.addEventListener('loadedmetadata',()=>seekObservation(moment.video_time_s),{once:true});
  }
}
function renderProductViews(){
  ui.teamsContent.replaceChildren();ui.developmentOverview.replaceChildren();ui.reportsContent.replaceChildren();
  if(api.mode==='production'){
    if(state.rosterLoading)ui.teamsContent.append(panel('Loading team roster','Checking authorized teams and players…'));
    else if(state.rosterError)ui.teamsContent.append(panel('Team roster unavailable',state.rosterError,[['Retry',loadRoster]]));
    else if(!state.teams?.length)ui.teamsContent.append(panel('No authorized teams','The backend returned no teams for this coach. Check school access or create a team through the approved backend workflow.'));
    for(const team of state.teams||[]){
      const teamId=team.id;
      if(!teamId)continue;
      const players=state.playersByTeam.get(teamId);
      ui.teamsContent.append(panel(team.name||teamId,`${players?.length??'—'} authorized players · backend record.`,[['View players',()=>loadPlayers(teamId)],['View sessions',()=>showView('library')]]));
      for(const player of players||[]){
        const playerId=player.id;
        if(!playerId)continue;
        const playerRuns=state.runs.filter(run=>String(run.player_id||'')===playerId);
        const actions=playerRuns.map((run,index)=>[`Open ${run.session_label||`Session ${index+1}`}`,()=>{state.focusPlayerId=playerId;loadSession(run.id);}]);
        actions.push(['View development',()=>{state.focusPlayerId=playerId;showView('development');}]);
        ui.teamsContent.append(panel(player.name||playerId,`${playerRuns.length} linked sessions returned by the backend. Run-local track IDs still require coach identity review.`,actions));
      }
    }
  }else{
    const teams=new Map();
    for(const run of state.runs){if(run.team_name){if(!teams.has(run.team_name))teams.set(run.team_name,[]);teams.get(run.team_name).push(run);}}
    if(!teams.size){ui.teamsContent.append(panel('Team roster unavailable','This Studio API has no authenticated school or team roster. Open a session to inspect run-scoped local player associations; temporary track IDs are not player identities.',[['Open sessions',()=>showView('library')]]));}
    for(const [team,runs] of teams){
      ui.teamsContent.append(panel(team,`${runs.length} linked ${runs.length===1?'session':'sessions'} · ${state.demo?'synthetic fixture':'team metadata supplied by run'}. No cross-session identity has been independently verified.`,[['View sessions',()=>showView('library')]]));
      const players=new Map();for(const run of runs)if(run.player_id!=null)players.set(String(run.player_id),run.player_name||`Player label ${run.player_id}`);
      for(const [playerId,name] of players){
        const playerRuns=runs.filter(run=>String(run.player_id)===playerId);
        const actions=playerRuns.map((run,index)=>[`Open ${run.session_label||`Session ${index+1}`}`,()=>{state.focusPlayerId=playerId;loadSession(run.id);}]);
        actions.push(['View development',()=>{state.focusPlayerId=playerId;showView('development');}]);
        ui.teamsContent.append(panel(name,`Local player label ${playerId} · ${playerRuns.length} linked sessions · ${state.demo?'fixture association':'identity review still required'}.`,actions));
      }
    }
  }
  const draft=state.developmentDraft;
  if(!draft){ui.developmentOverview.append(panel('Start with a reviewed moment','Open Session 1, select a coach-labelled local player association, choose a video moment, and record an observable instruction. This remains browser-local until an authenticated backend exists.',[['Open sessions',()=>showView('library')]]));}
  else{
    const steps=`Session 1: ${draft.first.run_id} at ${formatTime(draft.first.video_time_s)}. ${draft.followup?`Session 2: ${draft.followup.run_id} at ${formatTime(draft.followup.video_time_s)}.`:'Follow-up evidence has not been linked.'} ${draft.coach_outcome?`Coach outcome: ${draft.coach_outcome.replaceAll('_',' ')}.`:'Coach outcome pending.'}`;
    ui.developmentOverview.append(panel(`${draft.player_name||`Player label ${draft.player_id}`} · ${draft.focus}`,`${steps} Observable check: ${draft.observable_check}. This is a browser-local draft, not a verified cross-session identity or approved report.`,[['Open first evidence',()=>openEvidence(draft.first)],...(draft.followup?[['Open follow-up evidence',()=>openEvidence(draft.followup)]]:[]),['Edit in Session Studio',()=>openEvidence(draft.followup||draft.first)]]));
  }
  if(!draft?.coach_assessment||!draft.followup){ui.reportsContent.append(panel('No coach-approved report','A report requires two linked evidence moments and a coach-confirmed outcome. There is no authenticated approval or parent-sharing API in this Studio.',[['View development',()=>showView('development')]]));}
  else{
    const report=panel(`${draft.player_name||`Player label ${draft.player_id}`} · coach assessment`,'BROWSER-LOCAL DRAFT · not an approved report, verified identity, or parent-visible record.');
    report.classList.add('report-panel');
    const facts=document.createElement('dl');facts.className='report-facts';
    for(const [label,value] of [
      ['Development objective',draft.focus],['Observable check',draft.observable_check],
      ['Session 1 · instruction evidence',`${draft.first.run_id} at ${formatTime(draft.first.video_time_s)}`],
      ['Session 2 · follow-up evidence',draft.followup?`${draft.followup.run_id} at ${formatTime(draft.followup.video_time_s)}`:'Not linked'],
      ['Coach-confirmed outcome',draft.coach_outcome?.replaceAll('_',' ')||'Pending'],
      ['Coach assessment',draft.coach_assessment]
    ]){const dt=document.createElement('dt');dt.textContent=label;const dd=document.createElement('dd');dd.textContent=value;facts.append(dt,dd);}
    report.append(facts);
    for(const [label,action] of [['Open Session 1 evidence',()=>openEvidence(draft.first)],['Open Session 2 evidence',()=>openEvidence(draft.followup)],['Export local draft',()=>ui.btnExportCorrection.click()]]){
      const button=document.createElement('button');button.type='button';button.className='btn-ctrl';button.textContent=label;button.addEventListener('click',action);report.append(button);
    }
    ui.reportsContent.append(report);
  }
}
async function loadRoster(){
  showView('teams');
  if(api.mode!=='production')return;
  state.rosterLoading=true;state.rosterError=null;renderProductViews();
  try{state.teams=await api.getTeams();}
  catch(error){state.teams=null;state.rosterError=error.message;}
  finally{state.rosterLoading=false;renderProductViews();}
}
async function loadPlayers(teamId){
  if(api.mode!=='production')return;
  state.rosterLoading=true;renderProductViews();
  try{state.playersByTeam.set(teamId,await api.getPlayers(teamId));state.rosterError=null;}
  catch(error){state.rosterError=error.message;}
  finally{state.rosterLoading=false;renderProductViews();}
}
function libraryLoading() {
  ui.libraryState.hidden=true;ui.libraryList.replaceChildren();
  for(let i=0;i<3;i++){const row=document.createElement('div');row.className='library-skeleton';row.setAttribute('aria-hidden','true');ui.libraryList.append(row);}
}
function clearSession(message='Select a session to review.') {
  state.data=null; state.selected=null; state.calibrationVerified=false; state.frames.clear(); state.frameTimes=[]; state.lastRenderedFrame=null;state.localMarkers=[];state.timelineEntries=[];state.fixtureVideoOpen=false;state.mediaKind='source';
  state.metaActivities=[];state.metaConfigured=false;state.metaEnabled=false;state.metaRemaining=0;state.metaError=null;++state.metaPollToken;renderMetaActivity();
  ui.matchVideo.pause(); ui.matchVideo.removeAttribute('src'); ui.matchVideo.load();
  setFixturePresentation(false);
  ui.overlayCanvas.width=1; ui.overlayCanvas.height=1;
  ui.sessionTitle.textContent='No session selected'; ui.sessionMeta.textContent=message;
  ui.runStatus.textContent='No run'; ui.runStatus.className='state-pill';
  ui.calBadge.textContent='Pixel-space review'; ui.videoStatus.textContent='Select a session with a source video.';
  ui.videoStatus.hidden=false; ui.statObservations.textContent='—'; ui.statTracks.textContent='—';
  ui.statConfirmed.textContent='—'; ui.statPhysical.textContent='Unverified';
  notice(ui.coverageNotice,''); renderRecordingReview(); renderCues(); renderTracks(); renderTimeline(); renderDevelopment(); renderFrame();
}
function setFixturePresentation(showPlaceholder){
  ui.fixtureVideoPlaceholder.hidden=!showPlaceholder;
  ui.fixtureVideoPlaceholder.parentElement.classList.toggle('fixture-placeholder-active',showPlaceholder);
  ui.workspaceView.classList.toggle('fixture-empty',showPlaceholder);
  ui.fixtureInspectorPlaceholder.hidden=!showPlaceholder;
  ui.transportDeck.hidden=showPlaceholder;
  if(showPlaceholder){ui.videoStatus.hidden=true;ui.frameReadout.textContent='NO MATCH FOOTAGE';}
}
ui.btnOpenFixtureVideo.addEventListener('click',()=>{
  if(!state.demo||!state.data?.video_url)return;
  state.fixtureVideoOpen=true;setFixturePresentation(false);
  ui.matchVideo.src=state.data.video_url;ui.matchVideo.load();
  ui.videoStatus.textContent='Loading synthetic test pattern…';ui.videoStatus.hidden=false;
  if(window.__qaScenario==='two_sessions'&&state.frameTimes.length)ui.matchVideo.addEventListener('loadedmetadata',()=>seekObservation(state.frameTimes[0]),{once:true});
});
const plannedOutputs=[
  ['Tactical minimap','Pitch positions require a reviewed coordinate and calibration contract.'],
  ['MOTChallenge export','Track export needs a verified artifact from the recording worker.'],
  ['Jersey recognition','Jersey numbers are not verified player identities.'],
  ['Speed & distance','Physical measurements require independent calibration and validation.'],
  ['Automatic retraining','Model changes require a separate reviewed training workflow.'],
  ['Coach voice commentary','Automatic recording commentary is not connected to this analysis run.'],
  ['Advanced scoring','Performance scores need defined evidence and independent validation.'],
];
function rebuildEvidenceTimes(data){
  state.frames=new Map();
  for(const obs of safeArray(data.observations)){
    const normalized=obs?.frame?.timestamp_s;
    const original=TouchlineApi.sourceTimeFromFrameMap(data.frame_map,obs?.frame?.index);
    const t=state.mediaKind==='annotated' ? normalized :
      (Array.isArray(data.frame_map) ? original : api.mode==='production' ? null : normalized);
    if(!finite(t)||t<0||!Array.isArray(obs.bbox_xyxy)||obs.bbox_xyxy.length!==4)continue;
    if(!state.frames.has(t))state.frames.set(t,[]);
    state.frames.get(t).push(obs);
  }
  state.frameTimes=[...state.frames.keys()].sort((a,b)=>a-b);
}
function selectReviewVideo(kind){
  const data=state.data;
  if(!data||!(kind==='annotated'&&TouchlineApi.reviewSummary(data).annotatedVideoAvailable||
      kind==='source'&&data.video_url))return;
  state.mediaKind=kind;
  ui.matchVideo.pause();
  ui.matchVideo.src=kind==='annotated'?data.annotated_video_url:data.video_url;
  ui.matchVideo.load();
  rebuildEvidenceTimes(data);
  ui.videoStatus.textContent=kind==='annotated'?'Loading verified annotated output…':'Loading original source recording…';
  ui.videoStatus.hidden=false;
  renderRecordingReview();renderTimeline();renderFrame();
}
function renderRecordingReview(){
  if(!ui.recordingProvenance)return;
  ui.recordingArtifacts.replaceChildren();ui.recordingReviewAction.replaceChildren();
  if(!ui.plannedCapabilities.childElementCount)for(const [title,description] of plannedOutputs){const item=document.createElement('div');const strong=document.createElement('strong');strong.textContent=`${title} · Planned`;const detail=document.createElement('span');detail.textContent=description;item.append(strong,detail);ui.plannedCapabilities.append(item);}
  if(!state.data){ui.recordingProvenance.textContent='Open a run to inspect its available provenance and evidence.';return;}
  const data=state.data, summary=TouchlineApi.reviewSummary(data);
  const modeLabel=state.demo?'Synthetic demo fixture':api.mode==='local'?'Local Studio run':'Authenticated backend run';
  ui.recordingProvenance.textContent=`${modeLabel} · Run ${summary.runId || state.id} · ${summary.status.toUpperCase()}${summary.parentRunId?` · Child of ${summary.parentRunId}`:' · No verified parent run'}\nStop reason: ${data.manifest?.stop_reason || 'Not supplied'}\nOriginal video SHA-256: ${summary.sourceHash || 'Not supplied by this run contract'}\nNormalized video SHA-256: ${summary.normalizedHash || 'Not supplied by this run contract'}\nOriginal-to-normalized frame map: ${summary.frameMapAvailable?'Supplied; original times require exact map lookup':'Not supplied; original media timestamps cannot be verified here'}`;
  const entries=[
    [state.demo?'Demo fixture – no match footage uploaded':data.video_url?'Recording available':'Recording unavailable',state.demo?'Synthetic test pattern is available only through the explicit QA preview action.':data.video_url?'Source recording available; this is not an annotated export.':'No playable source artifact was returned.'],
    [state.demo?`${summary.observationCount} synthetic fixture observations`:summary.compatibleObservations?`${summary.observationCount} canonical observations available`:'Canonical observations unavailable',state.demo?'These example boxes and tracks are not field evidence or verified player identities.':summary.compatibleObservations?`Temporary track IDs can be reviewed. ${api.mode==='production'&&!summary.frameMapAvailable?'Source-video overlays are withheld until an exact frame map is supplied.':'Timeline positions use the evidence timestamps or exact frame-map correspondence.'}`:'Missing or incompatible touchline.observation.v1 evidence; experimental records cannot support coach identity review.'],
    [summary.annotatedVideoAvailable?'Annotated output available':'Annotated footage unavailable',summary.annotatedVideoAvailable?'Verified at local import; this overlay contains provisional track labels and motion predictions.':'No verified annotated-video artifact was returned for coach review.'],
    [state.demo?'Identity review is a demo state':'Identity review not connected',state.demo?'Fixture track labels are examples, not confirmed players.':'Confirmed roster association and saved child-run review require the backend contract.'],
  ];
  for(const [title,description] of entries){const item=document.createElement('div');const strong=document.createElement('strong');strong.textContent=title;const detail=document.createElement('span');detail.textContent=description;item.append(strong,detail);ui.recordingArtifacts.append(item);}
  if(summary.annotatedVideoAvailable&&!state.demo){
    for(const [kind,label] of [['annotated','View annotated output'],['source','View original source']]){
      const button=document.createElement('button');button.type='button';button.className='btn-ctrl';
      button.textContent=label;button.disabled=kind===state.mediaKind;
      button.addEventListener('click',()=>selectReviewVideo(kind));ui.recordingReviewAction.append(button);
    }
  }
  const message=document.createElement('p');
  message.textContent=state.selected!==null?`Track T#${state.selected} is a temporary local track. Confirm/reject and immutable child-run save require a reviewed roster, correction API, and persisted response.`:'Select a visible temporary track in the inspector to review its evidence. Confirm/reject remains unavailable until the backend correction contract is published.';
  const confirm=document.createElement('button');confirm.type='button';confirm.disabled=true;confirm.textContent='Confirm player association · unavailable';
  const reject=document.createElement('button');reject.type='button';reject.disabled=true;reject.textContent='Reject association · unavailable';
  ui.recordingReviewAction.append(message,confirm,reject);
}
async function loadSessions(preferred=null) {
  const token=++state.listToken;
  state.listLoading=true;
  if(state.view==='analysis')renderAnalysis();
  setBusy(true); notice(ui.sessionAlert,'');libraryLoading();
  try {
    const runs=await listReads.read('sessions');
    if(token!==state.listToken)return;
    if (!Array.isArray(runs)) throw new Error('Invalid session list');
    state.listError=null;state.runs=runs.filter(run=>run&&typeof run.id==='string');renderLibrary();
    ui.sessionSelect.replaceChildren();
    setOption(ui.sessionSelect,'', runs.length ? 'Select a session' : 'No sessions available');
    for (const run of state.runs) setOption(ui.sessionSelect,run.id,runLabel(run));
    const id=state.runs.some(r=>r.id===preferred) ? preferred : null;
    if (id) { ui.sessionSelect.value=id; await loadSession(id); }
    else if (state.id && state.runs.some(r=>r.id===state.id)) ui.sessionSelect.value=state.id;
    else if (!state.runs.length) { ++state.loadToken;detailReads.cancel();state.id=null; clearSession(api.mode==='production'?'No authorized sessions were returned.':'No local runs are available. Full-recording upload is not deployed.'); }
  } catch(error) {
    if(token!==state.listToken||error instanceof TouchlineRequest.CancelledRead)return;
    state.runs=[];state.listError=`Could not load sessions: ${error.message}. Use Refresh to retry.`;renderLibrary();
    notice(ui.sessionAlert,`Could not load sessions: ${error.message}. Use Refresh to retry.`);
    if (!state.data) {++state.loadToken;detailReads.cancel();clearSession('Session list unavailable.');}
  } finally { if(token===state.listToken){state.listLoading=false;setBusy(false);renderAnalysis();} }
}
async function loadSession(id) {
  const originView=state.view,token=++state.loadToken; state.id=id; state.selected=null;
  if (state.recorder?.state==='recording') state.recorder.stop();
  discardAudio();
  state.data=null;state.frames.clear();state.frameTimes=[];state.calibrationVerified=false;state.localMarkers=[];state.timelineEntries=[];state.fixtureVideoOpen=false;
  setFixturePresentation(false);
  ui.matchVideo.pause();ui.matchVideo.removeAttribute('src');ui.matchVideo.load();
  ui.overlayCanvas.width=1;ui.overlayCanvas.height=1;
  renderCues();renderTracks();renderTimeline();renderDevelopment();
  ui.videoStatus.textContent='Opening session…';ui.videoStatus.hidden=false;
  ui.runStatus.textContent='LOADING';ui.runStatus.className='state-pill';
  ui.sessionMeta.textContent='Fetching run evidence…';ui.calBadge.textContent='Checking calibration…';
  ui.statObservations.textContent='—';ui.statTracks.textContent='—';
  ui.statConfirmed.textContent='—';ui.statPhysical.textContent='Unverified';
  notice(ui.coverageNotice,'');notice(ui.sessionAlert,'');ui.sessionTitle.textContent='Loading session…';
  if(!ui.libraryView.hidden){ui.libraryState.hidden=false;ui.libraryState.textContent=`Opening ${id}…`;}
  try {
    const data=await detailReads.read(id);
    if (token!==state.loadToken) return;
    // The integrated Studio pages canonical observations after its first 1,200.
    // Keep the session generation guard so a late page cannot replace a newer run.
    if (api.mode==='local' && data.observations_next_offset!=null) {
      const rows=safeArray(data.observations);
      let offset=data.observations_next_offset;
      while(offset!=null && rows.length<200000) {
        const page=await api.getObservationPage(id,offset,Math.min(5000,200000-rows.length));
        if(token!==state.loadToken)return;
        rows.push(...page.observations);
        offset=page.next_offset;
      }
      data.observations=rows;
      data.observations_next_offset=offset;
    }
    state.data=data;
    refreshMetaActivity();
    state.localMarkers=readLocalMarkers(id);
    const observations=safeArray(data.observations);
    state.mediaKind=!state.demo&&TouchlineApi.reviewSummary(data).annotatedVideoAvailable?'annotated':'source';
    rebuildEvidenceTimes(data);
    ui.matchVideo.pause();
    if (state.demo && data.video_url) {
      setFixturePresentation(true);
    } else if ((state.mediaKind==='annotated'&&data.annotated_video_url)||data.video_url) {
      ui.matchVideo.src=state.mediaKind==='annotated'?data.annotated_video_url:data.video_url;
      ui.matchVideo.load();
      ui.videoStatus.textContent=state.mediaKind==='annotated'?'Loading verified annotated output…':'Loading source recording…'; ui.videoStatus.hidden=false;
    } else {
      ui.matchVideo.removeAttribute('src'); ui.matchVideo.load();
      ui.videoStatus.textContent='This session has no source video.'; ui.videoStatus.hidden=false;
    }
    const rawRunStatus=String(data.manifest?.status || 'unknown').toLowerCase();
    const runStatus=rawRunStatus==='completed'?'complete':rawRunStatus;
    ui.runStatus.textContent=runStatus.toUpperCase(); ui.runStatus.className=`state-pill ${runStatus}`;
    ui.sessionTitle.textContent=data.manifest?.source?.name || ui.sessionSelect.selectedOptions[0]?.textContent || id;
    ui.sessionMeta.textContent=state.demo?'Synthetic demo scenario · no match footage uploaded':`${data.manifest?.selected_frames ?? '—'} sampled frames · ${formatTime(data.analytics?.session?.duration_s)} observed span`;
    const cal=calibrationPayload(data);
    const verified=independentlyVerified(cal,data);state.calibrationVerified=verified;
    ui.calBadge.textContent=verified?'Calibrated pitch / verified':cal.status==='preliminary_unverified'?'Demo scale / unverified':cal.calibrated?'Scale / unverified':'Pixels mode / uncalibrated';
    ui.calBadge.className='status-badge';
    const total=data.manifest?.observation_count ?? data.analytics?.session?.observation_count;
    ui.statObservations.textContent=String(observations.length)+(finite(total) && total>observations.length ? ` / ${total}` : '');
    const roster=safeArray(data.analytics?.player_roster);
    ui.statTracks.textContent=String(roster.length);
    ui.statConfirmed.textContent=String(roster.filter(reviewedPlayer).length);
    ui.statPhysical.textContent=verified?'Verified scale':'Unverified';
    const warnings=[];
    if (runStatus==='partial') warnings.push(state.demo?'PARTIAL analysis · synthetic demo only. Example observations are not match evidence.':'PARTIAL analysis · only part of this recording has reviewable evidence.');
    if (runStatus==='failed') warnings.push('Analysis failed; displayed artifacts may be incomplete.');
    if (finite(total) && total>observations.length) warnings.push(`Only ${observations.length} of ${total} observations were loaded; overlays and track review are incomplete.`);
    if (api.mode==='production' && observations.length && !state.frameTimes.length) warnings.push('Original-video overlay is withheld: no compatible frame correspondence was supplied.');
    if (data.analytics?.synthetic && !state.demo) warnings.push('Synthetic fixture: not field evidence.');
    if (cal.calibrated && !verified) warnings.push('A physical scale is present in the data but independent calibration evidence is unavailable here; physical metrics are withheld.');
    notice(ui.coverageNotice,warnings.join(' '));
    renderRecordingReview(); renderCues(); renderTracks(); renderTimeline(); renderDevelopment(); ui.libraryState.hidden=true;showView('workspace');renderFrame();
  } catch(error) {
    if (token!==state.loadToken) return;
    clearSession('Session detail unavailable.');
    notice(ui.sessionAlert,`Could not load ${id}: ${error.message}. Choose the session again or Refresh.`);
    ui.libraryState.hidden=false;ui.libraryState.textContent=`Could not open ${id}: ${error.message}. Select it again or Refresh.`;
    showView(originView==='analysis'?'analysis':'library');
    if(originView==='analysis')ui.analysisState.textContent=`Could not open ${id}: ${error.message}. Refresh runs or choose another entry.`;
  }
}
ui.navOverview.addEventListener('click',()=>{renderOverview();showView('overview');});
ui.btnOverviewSessions.addEventListener('click',()=>showView('library'));
ui.navTeams.addEventListener('click',loadRoster);
ui.navDevelopment.addEventListener('click',()=>showView('development'));
ui.navReports.addEventListener('click',()=>showView('reports'));
ui.navLibrary.addEventListener('click',()=>showView('library'));
ui.navWorkspace.addEventListener('click',()=>showView('analysis'));
ui.btnAnalysisRefresh.addEventListener('click',()=>loadSessions());
ui.btnAnalysisSessions.addEventListener('click',()=>showView('library'));
ui.btnAnalysisDashboard.addEventListener('click',()=>showView('overview'));
ui.btnBackLibrary.addEventListener('click',()=>showView('library'));
ui.librarySearch.addEventListener('input',renderLibrary);

function videoTime() { return finite(ui.matchVideo.currentTime) ? ui.matchVideo.currentTime : 0; }
function observationTime() {
  // Decoder timestamps are elapsed from the first decoded frame. Browser media time normally has the same origin.
  const start=ui.matchVideo.seekable?.length ? ui.matchVideo.seekable.start(0) : 0;
  return videoTime()-start;
}
function sourceAnchorTime(){
  return state.mediaKind==='annotated'?
    TouchlineApi.sourceTimeFromPlayback(state.data?.frame_map,videoTime()):observationTime();
}
function seekObservation(time) {
  const start=ui.matchVideo.seekable?.length ? ui.matchVideo.seekable.start(0) : 0;
  ui.matchVideo.currentTime=Math.max(0,time+start); renderFrame();
}
function nearestFrame(time) {
  const arr=state.frameTimes; if (!arr.length) return null;
  let lo=0,hi=arr.length; while(lo<hi){const mid=(lo+hi)>>1;if(arr[mid]<time)lo=mid+1;else hi=mid;}
  const before=arr[Math.max(0,lo-1)],after=arr[Math.min(lo,arr.length-1)];
  const nearest=Math.abs(time-before)<=Math.abs(time-after)?before:after;
  const left=arr[Math.max(0,arr.indexOf(nearest)-1)],right=arr[Math.min(arr.length-1,arr.indexOf(nearest)+1)];
  const tolerance=Math.min(0.45,Math.max(0.04,(right-left)/2));
  return Math.abs(nearest-time)<=tolerance ? nearest : null;
}
function syncCanvas(frame) {
  const video=ui.matchVideo, canvas=ui.overlayCanvas, box=video.parentElement;
  const vw=video.videoWidth,vh=video.videoHeight;
  if (!vw || !vh || !box.clientWidth || !box.clientHeight) return false;
  const scale=Math.min(box.clientWidth/vw,box.clientHeight/vh);
  const width=vw*scale,height=vh*scale;
  canvas.style.width=`${width}px`; canvas.style.height=`${height}px`;
  canvas.style.left=`${(box.clientWidth-width)/2}px`; canvas.style.top=`${(box.clientHeight-height)/2}px`;
  const source=state.data?.source_geometry;
  const manifestShape=state.data?.manifest?.image_shape_hw;
  const analysisPolicy=state.data?.manifest?.source?.analysis_policy;
  const pair=Array.isArray(source)&&source.length===2?source:
    Array.isArray(manifestShape)&&manifestShape.length===2?[manifestShape[1],manifestShape[0]]:
    finite(analysisPolicy?.width)&&finite(analysisPolicy?.height)?[analysisPolicy.width,analysisPolicy.height]:null;
  const fw=pair?.[0] ?? frame?.width ?? state.data?.analytics?.session?.frame_width;
  const fh=pair?.[1] ?? frame?.height ?? state.data?.analytics?.session?.frame_height;
  const frameMismatch=pair&&finite(frame?.width)&&finite(frame?.height)&&
    (frame.width!==fw||frame.height!==fh);
  if (!finite(fw)||!finite(fh)||fw<=0||fh<=0||frameMismatch||Math.abs(fw/fh-vw/vh)>0.01) {
    canvas.width=1;canvas.height=1;
    if (!state.geometryWarning) { state.geometryWarning=true; notice(ui.sessionAlert,'Overlay hidden: observation geometry is missing or does not match the video.'); }
    return false;
  }
  ui.frameReadout.textContent=`SOURCE PIXELS / ${fw} × ${fh}`;
  if (canvas.width!==fw||canvas.height!==fh) {canvas.width=fw;canvas.height=fh;}
  return true;
}
function trackColor(id) {
  const colors=['#4cc9f0','#64d9aa','#f4c76d','#db9aff','#ff9b85','#8ca8ff'];
  const n=Number(id); return colors[Math.abs(Number.isFinite(n)?Math.trunc(n):0)%colors.length];
}
function rosterFor(id) {return safeArray(state.data?.analytics?.player_roster).find(p=>String(p.local_track_id)===String(id));}
function allCues() {return [...safeArray(state.data?.transcript?.segments),...safeArray(state.data?.transcript?.manual_cues)].filter(s=>finite(s.video_start_s)).sort((a,b)=>a.video_start_s-b.video_start_s);}
function markerStorageKey(id){return `touchline.local_markers.v1.${encodeURIComponent(id)}`;}
function readLocalMarkers(id){try{const value=JSON.parse(localStorage.getItem(markerStorageKey(id))||'[]');return safeArray(value).filter(x=>x&&x.session_id===id);}catch{toast('Local markers could not be read; browser storage was not changed.','error');return [];}}
function updateLocalMarkerTicks(){
  ui.timelineMarkers.querySelectorAll('.local-timeline-marker').forEach(el=>el.remove());
  const duration=ui.matchVideo.duration||state.data?.analytics?.session?.duration_s;
  if(!finite(duration)||duration<=0)return;
  for(const marker of state.localMarkers){if(!TouchlineTimeline.validTime(marker.timestamp_s)||marker.timestamp_s>duration)continue;
    const tick=document.createElement('button');tick.type='button';tick.className='timeline-marker local-timeline-marker';tick.style.left=`${Math.max(0,Math.min(100,marker.timestamp_s/duration*100))}%`;tick.title=`Local marker at ${formatTime(marker.timestamp_s)}`;tick.setAttribute('aria-label',tick.title);tick.addEventListener('click',()=>seekObservation(marker.timestamp_s));ui.timelineMarkers.append(tick);
  }
}
function renderTimeline(){
  state.timelineEntries=state.data ? TouchlineTimeline.project(state.id,state.data,state.localMarkers) : [];
  ui.timelineCount.textContent=String(state.timelineEntries.length);ui.timelineList.replaceChildren();
  updateLocalMarkerTicks();
  if(!state.timelineEntries.length){const empty=document.createElement('p');empty.className='empty-state';empty.textContent='No timestamped cues or markers in this run.';ui.timelineList.append(empty);return;}
  for(const item of state.timelineEntries){
    const row=document.createElement('div');row.className='timeline-entry';row.dataset.start=String(item.start_s);row.dataset.end=String(item.end_s);
    const jump=document.createElement('button');jump.type='button';jump.className='timeline-entry-main';jump.setAttribute('aria-label',`Jump to ${formatTime(item.start_s)}: ${item.text}`);
    const label=document.createElement('span');label.className='timeline-entry-label';label.textContent=`${formatTime(item.start_s)} / ${item.source.replaceAll('_',' ').toUpperCase()}${item.local_only?' / LOCAL ONLY':''}`;
    const body=document.createElement('strong');body.textContent=item.text;
    const detail=document.createElement('small');
    detail.textContent=[item.speaker_label?`Speaker ${item.speaker_label}`:null,item.player_id!=null?`Player reference ${item.player_id} / verify association`:item.local_track_id!=null?`Temporary track T#${item.local_track_id}`:null,item.source==='ai_observation'?'AI draft / verify evidence':null,item.source==='detected_event'?'Detected event / verify evidence':null].filter(Boolean).join(' · ');
    jump.append(label,body);if(detail.textContent)jump.append(detail);jump.addEventListener('click',()=>{ui.matchVideo.pause();seekObservation(item.start_s);});row.append(jump);
    if(item.local_only){const remove=document.createElement('button');remove.type='button';remove.className='btn-mini timeline-remove';remove.textContent='Remove local marker';remove.addEventListener('click',()=>removeLocalMarker(item.id));row.append(remove);}
    ui.timelineList.append(row);
  }
}
function removeLocalMarker(id){
  const next=state.localMarkers.filter(x=>String(x.marker_id)!==id);
  try{localStorage.setItem(markerStorageKey(state.id),JSON.stringify(next));state.localMarkers=next;renderTimeline();toast('Local marker removed.','success');}
  catch(error){toast(`Marker could not be removed: ${error.message}`,'error');}
}
function reviewedRoster(){return safeArray(state.data?.analytics?.player_roster).filter(reviewedPlayer);}
function renderDevelopment(){
  const previous=ui.developmentPlayer.value,roster=reviewedRoster(),draft=state.developmentDraft;
  ui.overviewDraftStatus.textContent=draft?'1 browser-local draft':'No local draft';
  ui.developmentPlayer.replaceChildren();setOption(ui.developmentPlayer,'',roster.length?'Choose a local player label':'No coach-labelled association in this run');
  for(const p of roster)setOption(ui.developmentPlayer,String(p.confirmed_player_id),`${p.player_name||`Player ${p.confirmed_player_id}`} · local association only`);
  if(roster.some(p=>String(p.confirmed_player_id)===(draft?.player_id||state.focusPlayerId||previous)))ui.developmentPlayer.value=draft?.player_id||state.focusPlayerId||previous;
  ui.developmentPlayer.disabled=!!draft||!roster.length;
  ui.developmentFocus.disabled=!!draft;ui.developmentCheck.disabled=!!draft;
  ui.btnCreateCorrection.disabled=!!draft||!roster.length;
  ui.btnAddFollowup.disabled=!draft||!!draft.followup||!state.data||state.id===draft.first.run_id||!TouchlineDevelopment.confirmedPlayer(roster,draft.player_id);
  ui.btnDiscardCorrection.disabled=!draft;
  ui.developmentAssessment.disabled=!draft?.followup;
  ui.developmentOutcome.disabled=!draft?.followup;
  ui.btnSaveAssessment.disabled=!draft?.followup;
  ui.btnExportCorrection.disabled=!draft?.coach_assessment;
  ui.developmentDraft.replaceChildren();
  ui.developmentComparison.replaceChildren();
  ui.developmentComparison.hidden=!draft?.followup;
  if(!draft){ui.developmentDraft.textContent=roster.length?'No local draft. Choose a player label and a video moment.':'No local draft. This run has no coach-labelled player association.';return;}
  const title=document.createElement('strong');title.textContent=`${draft.player_name||`Player ${draft.player_id}`} · ${draft.focus}`;
  const check=document.createElement('p');check.textContent=`Observable check: ${draft.observable_check}`;
  const status=document.createElement('p');status.className='muted';status.textContent='BROWSER-LOCAL COACH DRAFT · not authenticated or parent-visible';
  ui.developmentDraft.append(title,check,status);
  for(const [label,evidence] of [['First evidence',draft.first],['Follow-up evidence',draft.followup]]){
    if(!evidence)continue;
    const jump=document.createElement('button');jump.type='button';jump.className='btn-mini development-evidence';jump.textContent=`${label}: ${evidence.run_id} at ${formatTime(evidence.video_time_s)} — open moment`;
    jump.addEventListener('click',async()=>{
      if(state.id!==evidence.run_id)await loadSession(evidence.run_id);
      if(state.id===evidence.run_id&&state.data){if(ui.matchVideo.readyState>=1)seekObservation(evidence.video_time_s);else ui.matchVideo.addEventListener('loadedmetadata',()=>seekObservation(evidence.video_time_s),{once:true});}
    });ui.developmentDraft.append(jump);
  }
  if(draft.coach_assessment){const assessment=document.createElement('p');assessment.textContent=`Coach outcome: ${(draft.coach_outcome||'unclassified').replaceAll('_',' ')} · ${draft.coach_assessment}`;ui.developmentDraft.append(assessment);}
  if(draft.followup){
    const title=document.createElement('strong');title.textContent='SESSION COMPARISON / COACH REVIEW';
    const grid=document.createElement('div');grid.className='comparison-grid';
    for(const [label,moment] of [['Session 1 · instruction',draft.first],['Session 2 · follow-up',draft.followup]]){
      const card=document.createElement('button');card.type='button';card.className='comparison-card';
      const name=document.createElement('strong');name.textContent=label;
      const detail=document.createElement('span');detail.textContent=`${moment.run_id} · ${formatTime(moment.video_time_s)} · open video moment`;
      card.append(name,detail);card.addEventListener('click',()=>openEvidence(moment));grid.append(card);
    }
    const note=document.createElement('p');note.className='muted';note.textContent='These are coach-selected moments. AI reassessment is unavailable because this Studio has no authenticated evidence-grounded AI endpoint. Review both clips before confirming an outcome.';
    ui.developmentComparison.append(title,grid,note);
  }
  ui.developmentOutcome.value=draft.coach_outcome||'';
  if(ui.developmentAssessment.value!==draft.coach_assessment&&draft.coach_assessment)ui.developmentAssessment.value=draft.coach_assessment;
}
function renderFrame() {
  if(state.demo&&!state.fixtureVideoOpen)return;
  const time=observationTime(), duration=ui.matchVideo.duration;
  ui.videoSlider.max=finite(duration)?duration:Math.max(state.frameTimes.at(-1)||0,1);
  ui.videoSlider.value=videoTime(); ui.timeDisplay.textContent=`${formatTime(videoTime())} / ${formatTime(duration)}`;
  const cue=allCues().find(s=>time>=s.video_start_s&&time<=(finite(s.video_end_s)?s.video_end_s:s.video_start_s+2.5));
  ui.videoSubtitlesOverlay.classList.toggle('active',!!cue&&ui.chkSubsOverlay.checked);
  ui.subtitleText.textContent=cue?.text || '';
  for(const card of ui.subtitlesList.querySelectorAll('.subtitle-card')) {
    card.classList.toggle('active-cue',time>=Number(card.dataset.start)&&time<=Number(card.dataset.end));
  }
  for(const card of ui.timelineList.querySelectorAll('.timeline-entry'))card.classList.toggle('active-cue',time>=Number(card.dataset.start)&&time<=Number(card.dataset.end));
  const frameTime=nearestFrame(time), observations=frameTime==null?[]:state.frames.get(frameTime);
  const frame=observations?.[0]?.frame;
  // The reviewed export already burns provisional tracks into a resized video.
  // Canonical boxes use normalized source pixels and must not be drawn on it.
  if(state.mediaKind==='annotated'){
    ui.overlayCanvas.style.display='none';
    ui.frameReadout.textContent='ANNOTATED OUTPUT / PROVISIONAL TRACKS';
    return;
  }
  ui.overlayCanvas.style.display='';
  if (!syncCanvas(frame)) return;
  ctx.clearRect(0,0,ui.overlayCanvas.width,ui.overlayCanvas.height);
  if (!observations) return;
  if (ui.chkTrails.checked) {
    for(const p of safeArray(state.data?.analytics?.player_roster)) {
      if(state.selected!=null&&String(state.selected)!==String(p.local_track_id))continue;
      const pts=safeArray(p.trajectory).filter(pt=>finite(pt.timestamp_s)&&pt.timestamp_s<=time&&Array.isArray(pt.ground_contact_px));
      if(pts.length<2)continue;
      ctx.beginPath();ctx.strokeStyle=trackColor(p.local_track_id);ctx.globalAlpha=.6;ctx.lineWidth=2;
      pts.forEach((pt,i)=>{const [x,y]=pt.ground_contact_px;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();ctx.globalAlpha=1;
    }
  }
  if (!ui.chkBoxes.checked) return;
  for(const obs of observations) {
    const id=obs.local_track_id;if(state.selected!=null&&String(id)!==String(state.selected))continue;
    const box=obs.bbox_xyxy;if(!box.every(finite))continue;
    const [x1,y1,x2,y2]=box;if(x2<=x1||y2<=y1)continue;
    const color=trackColor(id),p=rosterFor(id);
    ctx.strokeStyle=color;ctx.lineWidth=state.selected!=null?3:2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    if(ui.chkTags.checked){
      const tag=reviewedPlayer(p)?`Local label ${p.confirmed_player_id} · T${id}`:`Track ${id}`;
      ctx.font='bold 13px system-ui';const tw=ctx.measureText(tag).width+12;
      const ty=Math.max(0,y1-23);ctx.fillStyle=color;ctx.fillRect(x1,ty,tw,22);
      ctx.fillStyle='#0a1119';ctx.fillText(tag,x1+6,ty+15);
    }
  }
}
function renderCues() {
  ui.subtitlesList.replaceChildren();ui.timelineMarkers.replaceChildren();
  const cues=allCues();ui.subsCount.textContent=cues.length;
  if(!cues.length){const p=document.createElement('p');p.className='empty-state';p.textContent='No transcript or coaching cues yet. Record audio or add a manual cue.';ui.subtitlesList.append(p);return;}
  for(const [index,cue] of cues.entries()) {
    const card=document.createElement('article');card.className='subtitle-card';
    card.dataset.start=cue.video_start_s;card.dataset.end=finite(cue.video_end_s)?cue.video_end_s:cue.video_start_s+2.5;
    const head=document.createElement('div');head.className='cue-header';
    head.textContent=`${formatTime(cue.video_start_s)}–${formatTime(cue.video_end_s)} · ${cue.speaker_label ? `${cue.speaker_label} (unverified)` : 'Coach cue'} · ${index+1}`;
    const body=document.createElement('p');body.className='cue-body';body.textContent=cue.text||'—';
    const actions=document.createElement('div');actions.className='cue-actions';
    for(const [label,callback] of [['Go to time',()=>seekObservation(cue.video_start_s)],['Edit',()=>openCue(cue)],['Delete',()=>deleteCue(cue)]]){
      const button=document.createElement('button');button.type='button';button.className='btn-mini';button.textContent=label;button.addEventListener('click',callback);actions.append(button);
    }
    card.append(head,body,actions);ui.subtitlesList.append(card);
    const duration=ui.matchVideo.duration||state.data?.analytics?.session?.duration_s;
    if(finite(duration)&&duration>0){const marker=document.createElement('button');marker.type='button';marker.className='timeline-marker';marker.style.left=`${Math.max(0,Math.min(100,cue.video_start_s/duration*100))}%`;marker.title=`Cue at ${formatTime(cue.video_start_s)}`;marker.setAttribute('aria-label',marker.title);marker.addEventListener('click',()=>seekObservation(cue.video_start_s));ui.timelineMarkers.append(marker);}
  }
}
function renderTracks(){
  const roster=safeArray(state.data?.analytics?.player_roster);ui.playersList.replaceChildren();ui.playerCount.textContent=roster.length;
  if(!roster.length){const p=document.createElement('p');p.className='empty-state';p.textContent='No tracks in the loaded analysis.';ui.playersList.append(p);}
  const selected=rosterFor(state.selected);
  const verified=state.calibrationVerified;
  ui.selectedTrackInfo.textContent=selected ? `Track T#${selected.local_track_id} · ${reviewedPlayer(selected)?`local player label #${selected.confirmed_player_id}; server identity unverified`:'identity unconfirmed'} · ${selected.observation_count ?? safeArray(selected.trajectory).length} observations${selected.tracking_quality?` · ${selected.tracking_quality} tracking`:''}. ${verified?'Server-reported independent calibration is available.':'Physical workload remains unverified.'}` : 'Select a temporary track to inspect available observations. Track IDs are not player identities.';
  renderRecordingReview();
  for(const p of roster){
    const row=document.createElement('button');row.type='button';row.className='player-card';
    row.classList.toggle('selected',String(state.selected)===String(p.local_track_id));
    row.setAttribute('aria-pressed',row.classList.contains('selected'));
    const title=document.createElement('strong');title.textContent=reviewedPlayer(p)?`Local label #${p.confirmed_player_id} ${p.player_name||''} (T#${p.local_track_id}) · not server verified`:`Track T#${p.local_track_id} · unconfirmed`;
    title.style.color=trackColor(p.local_track_id);
    const meta=document.createElement('small');
    const movement=p.movement||{};
    const speed=verified&&finite(movement.average_speed_mps)?`${movement.average_speed_mps} m/s`:
      finite(movement.average_pixel_speed_px_per_s)?`${movement.average_pixel_speed_px_per_s} px/s`:'Speed —';
    const distance=verified&&finite(movement.distance_m)?`${movement.distance_m} m`:
      finite(movement.image_pixel_distance_px)?`${movement.image_pixel_distance_px} px`:'Distance —';
    meta.textContent=`${speed} · ${distance} · ${safeArray(p.trajectory).length} points`;
    row.append(title,meta);row.addEventListener('click',()=>{state.selected=String(state.selected)===String(p.local_track_id)?null:p.local_track_id;renderTracks();renderFrame();});ui.playersList.append(row);
  }
}

ui.btnRetrySessions.addEventListener('click',()=>loadSessions());
ui.sessionSelect.addEventListener('change',()=>{if(ui.sessionSelect.value){const run=state.runs.find(item=>item.id===ui.sessionSelect.value);if(run)openRunOrJob(run);}else{++state.loadToken;detailReads.cancel();state.id=null;clearSession();}});
ui.matchVideo.addEventListener('loadedmetadata',()=>{ui.videoStatus.hidden=true;state.geometryWarning=false;renderCues();renderTimeline();renderFrame();});
ui.matchVideo.addEventListener('error',()=>{ui.videoStatus.textContent='Video could not be loaded. Check the source artifact.';ui.videoStatus.hidden=false;});
ui.matchVideo.addEventListener('timeupdate',renderFrame);
ui.matchVideo.addEventListener('play',()=>{ui.btnPlayPause.textContent='Pause';});
ui.matchVideo.addEventListener('pause',()=>{ui.btnPlayPause.textContent='Play';renderFrame();});
ui.matchVideo.addEventListener('ended',()=>{if(ui.chkLoop.checked){ui.matchVideo.currentTime=0;ui.matchVideo.play().catch(()=>{});}});
ui.btnPlayPause.addEventListener('click',()=>{if(!ui.matchVideo.src)return;if(ui.matchVideo.paused)ui.matchVideo.play().catch(e=>toast(e.message,'error'));else ui.matchVideo.pause();});
function step(direction){const t=observationTime(),times=state.frameTimes;if(!times.length)return;ui.matchVideo.pause();const target=direction>0?times.find(x=>x>t+.001):times.findLast(x=>x<t-.001);if(finite(target))seekObservation(target);}
ui.btnStepBack.addEventListener('click',()=>step(-1));ui.btnStepFwd.addEventListener('click',()=>step(1));
ui.videoSlider.addEventListener('input',()=>{ui.matchVideo.pause();ui.matchVideo.currentTime=Number(ui.videoSlider.value);renderFrame();});
ui.speedSelect.addEventListener('change',()=>{ui.matchVideo.playbackRate=Number(ui.speedSelect.value)||1;});
for(const id of ['chkBoxes','chkTags','chkTrails','chkSubsOverlay'])ui[id].addEventListener('change',renderFrame);
window.addEventListener('resize',renderFrame);
window.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement?.tagName)||document.querySelector('.modal-backdrop[style*="flex"]'))return;
  if(event.code==='Space'){event.preventDefault();ui.btnPlayPause.click();}else if(event.code==='ArrowLeft'){event.preventDefault();step(-1);}else if(event.code==='ArrowRight'){event.preventDefault();step(1);}
});
ui.overlayCanvas.addEventListener('click',event=>{
  const rect=ui.overlayCanvas.getBoundingClientRect(),x=(event.clientX-rect.left)*ui.overlayCanvas.width/rect.width,y=(event.clientY-rect.top)*ui.overlayCanvas.height/rect.height;
  const t=nearestFrame(observationTime());if(t==null)return;
  const hit=state.frames.get(t).findLast(o=>Array.isArray(o.bbox_xyxy)&&x>=o.bbox_xyxy[0]&&x<=o.bbox_xyxy[2]&&y>=o.bbox_xyxy[1]&&y<=o.bbox_xyxy[3]);
  if(hit){state.selected=hit.local_track_id;renderTracks();renderFrame();}
});
for(const tab of document.querySelectorAll('.tab-item'))tab.addEventListener('click',()=>{
  for(const item of document.querySelectorAll('.tab-item')){const active=item===tab;item.classList.toggle('active',active);item.setAttribute('aria-selected',String(active));item.tabIndex=active?0:-1;$(item.dataset.tab).style.display=active?'flex':'none';}
});
document.querySelector('.sidebar-tabs').addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  const tabs=[...document.querySelectorAll('.tab-item')],index=tabs.indexOf(document.activeElement);if(index<0)return;
  event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[next].focus();tabs[next].click();
});

async function refreshDevices(){
  ui.micInputSelect.replaceChildren();setOption(ui.micInputSelect,'','Default microphone');
  if(!navigator.mediaDevices?.enumerateDevices){notice(ui.micStatusLabel,'Microphone device selection unavailable in this browser.');return;}
  try{const devices=await navigator.mediaDevices.enumerateDevices();let index=0;for(const device of devices.filter(d=>d.kind==='audioinput'))setOption(ui.micInputSelect,device.deviceId,device.label||`Microphone ${++index}`);}
  catch(error){ui.micStatusLabel.textContent=`Cannot list microphones: ${error.message}`;}
}
function discardAudio(){if(state.audioUrl)URL.revokeObjectURL(state.audioUrl);state.audioUrl=null;state.recorded=null;ui.audioPreview.removeAttribute('src');ui.audioReview.hidden=true;ui.audioFileInput.value='';}
function stopStream(){state.stream?.getTracks().forEach(t=>t.stop());state.stream=null;}
function recordButton(label){
  const dot=document.createElement('span');dot.className='record-dot';dot.setAttribute('aria-hidden','true');
  ui.btnRecordMic.replaceChildren(dot,document.createTextNode(label));
}
ui.btnRefreshDevices.addEventListener('click',refreshDevices);
ui.btnDiscardAudio.addEventListener('click',()=>{discardAudio();ui.micStatusLabel.textContent='Recording discarded.';});
async function uploadRecordedAudio() {
  const rec=state.recorded;
  if(!rec||state.uploadingAudio)return;
  if(state.id!==rec.runId){toast('This recording belongs to another session.','error');return;}
  state.uploadingAudio=true;ui.btnUploadAudio.disabled=true;ui.btnDiscardAudio.disabled=true;ui.btnRecordMic.disabled=true;
  ui.micStatusLabel.textContent='Uploading completed audio for transcription…';
  try{
    if(api.mode!=='local'||state.demo)throw new Error('Muse Voice requires a real local Studio session.');
    const data=await jsonResponse(await fetch(`${runUrl(rec.runId)}/meta-audio?anchor_s=${encodeURIComponent(rec.anchor)}`,
      {method:'POST',headers:{'Content-Type':rec.blob.type||'application/octet-stream'},body:rec.blob}));
    if(data.status!=='submitted'||!data.activity_id)throw new Error('Meta did not accept an audio job.');
    discardAudio();ui.micStatusLabel.textContent='Audio submitted to Muse Voice. Review the actual result in Meta model activity; it is not yet a coach-approved transcript.';
    refreshMetaActivity();toast('Audio submitted for transcription.','success');
  }catch(error){
    if(state.id===rec.runId){ui.micStatusLabel.textContent=`Transcription failed: ${error.message}. Recording kept for retry.`;toast(error.message,'error');}
  }finally{state.uploadingAudio=false;ui.btnUploadAudio.disabled=false;ui.btnDiscardAudio.disabled=false;ui.btnRecordMic.disabled=false;}
}
ui.btnRecordMic.addEventListener('click',async()=>{
  if(state.recorder?.state==='recording'){ui.matchVideo.pause();state.recorder.stop();return;}
  if(!state.id||!state.data?.video_url){toast('Open a session with source video first.','error');return;}
  if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){toast('Browser audio recording unavailable.','error');return;}
  discardAudio();
  const requestedRun=state.id;
  try{
    const device=ui.micInputSelect.value;
    ui.micStatusLabel.textContent='Requesting microphone access…';
    const stream=await navigator.mediaDevices.getUserMedia({audio:device?{deviceId:{exact:device}}:true});
    if(state.id!==requestedRun){stream.getTracks().forEach(t=>t.stop());return;}
    state.stream=stream;
    const chunks=[];
    const anchor=sourceAnchorTime();
    if(!Number.isFinite(anchor)){stopStream();ui.micStatusLabel.textContent='Original source time is unavailable. Switch to source video before recording coach audio.';return;}
    const recording={runId:requestedRun,anchor,monotonic:performance.now(),language:ui.headerLangSelect.value,failed:false};
    state.recording=recording;
    const recorder=new MediaRecorder(stream);state.recorder=recorder;
    recorder.ondataavailable=e=>{if(e.data?.size)chunks.push(e.data);};
    recorder.onerror=e=>{recording.failed=true;ui.micStatusLabel.textContent=`Recording error: ${e.error?.message||'unknown'}`;stopStream();ui.matchVideo.pause();};
    recorder.onstop=()=>{
      stopStream();clearInterval(state.clock);state.clock=null;
      recordButton(' Record collar mic');
      ui.btnRecordMic.classList.remove('recording');
      if(recording.failed||state.disposing||state.id!==recording.runId)return;
      const blob=new Blob(chunks,{type:recorder.mimeType||'audio/webm'});
      if(!blob.size){ui.micStatusLabel.textContent='No audio was captured.';return;}
      state.recorded={blob,...recording};state.audioUrl=URL.createObjectURL(blob);
      ui.audioPreview.src=state.audioUrl;ui.audioReview.hidden=false;
      ui.micStatusLabel.textContent='Audio captured. Review it, then choose Transcribe with Muse Voice.';
    };
    await ui.matchVideo.play();
    if(state.id!==requestedRun){ui.matchVideo.pause();stopStream();return;}
    recorder.start(250);recordButton(' Stop & transcribe');
    ui.btnRecordMic.classList.add('recording');ui.micStatusLabel.textContent='Recording against video playback. Stop to transcribe.';
    state.clock=setInterval(()=>{ui.recordElapsed.textContent=formatTime((performance.now()-recording.monotonic)/1000).slice(0,5);},250);
    refreshDevices();
  }catch(error){stopStream();ui.matchVideo.pause();ui.micStatusLabel.textContent=`Recording could not start: ${error.message}`;}
});
ui.btnUploadAudio.addEventListener('click',uploadRecordedAudio);
ui.audioFileInput.addEventListener('change',()=>{
  const file=ui.audioFileInput.files?.[0];if(!file)return;
  if(!state.id||!state.data?.video_url){ui.micStatusLabel.textContent='Open a real session before selecting coach audio.';ui.audioFileInput.value='';return;}
  if(file.size<100||file.size>8_000_000){ui.micStatusLabel.textContent='Choose an audio file between 100 bytes and 8 MB.';ui.audioFileInput.value='';return;}
  const supported=['audio/wav','audio/webm','audio/ogg','audio/mp4','audio/x-wav'];
  if(file.type&&!supported.includes(file.type)){ui.micStatusLabel.textContent='Choose WAV, WebM, Ogg or MP4 audio.';ui.audioFileInput.value='';return;}
  if(state.audioUrl)URL.revokeObjectURL(state.audioUrl);
  const anchor=sourceAnchorTime();
  if(!Number.isFinite(anchor)){ui.micStatusLabel.textContent='Original source time is unavailable. Switch to source video before attaching coach audio.';ui.audioFileInput.value='';return;}
  state.recorded={blob:file,runId:state.id,anchor};state.audioUrl=URL.createObjectURL(file);
  ui.audioPreview.src=state.audioUrl;ui.audioReview.hidden=false;
  ui.micStatusLabel.textContent='Selected coach audio is ready. Review it before sending one bounded Meta transcription request.';
});

function openCue(cue=null){if(!state.data){toast('Select a session first.','error');return;}state.editing=cue;
  ui.modalSubTitle.textContent=cue?'Edit coach cue':'Add coach cue';ui.subStartInput.value=cue?.video_start_s??observationTime();
  ui.subEndInput.value=cue?.video_end_s??(observationTime()+2.5);ui.subTextInput.value=cue?.text||'';
  ui.subLangSelect.value=cue?.language_code||ui.headerLangSelect.value;ui.saveStatus.textContent='';showModal(ui.subtitleModal,ui.subTextInput);}
ui.btnAddManualSubtitle.addEventListener('click',()=>openCue());
ui.btnAddMarker.addEventListener('click',()=>{if(!state.id||!state.data){toast('Open a session first.','error');return;}ui.markerTime.value=observationTime().toFixed(2);ui.markerLabel.value='';ui.markerStatus.textContent='';showModal(ui.markerModal,ui.markerLabel);});
ui.btnCancelMarker.addEventListener('click',()=>hideModal(ui.markerModal));
ui.btnSaveMarker.addEventListener('click',()=>{
  const timestamp_s=Number(ui.markerTime.value),label=ui.markerLabel.value.trim();
  const duration=ui.matchVideo.duration;
  if(!TouchlineTimeline.validTime(timestamp_s)||!label||!finite(duration)||duration<=0||timestamp_s>duration){ui.markerStatus.textContent='Enter a note and a video time within this recording.';return;}
  const marker={marker_id:crypto.randomUUID(),session_id:state.id,timestamp_s,label,created_at:new Date().toISOString()};
  const next=[...state.localMarkers,marker];
  try{localStorage.setItem(markerStorageKey(state.id),JSON.stringify(next));state.localMarkers=next;renderTimeline();hideModal(ui.markerModal);toast('Local marker saved in this browser.','success');}
  catch(error){ui.markerStatus.textContent=`Marker was not saved: ${error.message}`;}
});
ui.btnCreateCorrection.addEventListener('click',()=>{
  const timeS=observationTime(),duration=ui.matchVideo.duration,playerId=ui.developmentPlayer.value;
  if(!state.data||!finite(duration)||duration<=0||timeS>duration){ui.developmentStatus.textContent='Load playable video and choose a moment within its duration.';return;}
  const player=reviewedRoster().find(p=>String(p.confirmed_player_id)===playerId);
  try{state.developmentDraft=TouchlineDevelopment.start({id:crypto.randomUUID(),playerId,playerName:player?.player_name||'',focus:ui.developmentFocus.value,observableCheck:ui.developmentCheck.value,runId:state.id,timeS,roster:reviewedRoster()});ui.developmentStatus.textContent='First moment linked in a browser-local draft. Matching labels do not verify identity.';persistDevelopment();renderDevelopment();}
  catch(error){ui.developmentStatus.textContent=error.message;}
});
ui.btnAddFollowup.addEventListener('click',()=>{
  const timeS=observationTime(),duration=ui.matchVideo.duration;
  if(!finite(duration)||duration<=0||timeS>duration){ui.developmentStatus.textContent='Choose a moment within the playable video.';return;}
  try{state.developmentDraft=TouchlineDevelopment.addFollowup(state.developmentDraft,{runId:state.id,timeS,roster:reviewedRoster()});ui.developmentStatus.textContent='Two moments linked. Compare both clips, then record only what the coach observes.';persistDevelopment();renderDevelopment();}
  catch(error){ui.developmentStatus.textContent=error.message;}
});
ui.btnSaveAssessment.addEventListener('click',()=>{
  try{state.developmentDraft=TouchlineDevelopment.assess(state.developmentDraft,ui.developmentAssessment.value,ui.developmentOutcome.value);ui.developmentStatus.textContent='Coach outcome saved in this browser only. No parent report was published.';persistDevelopment();renderDevelopment();}
  catch(error){ui.developmentStatus.textContent=error.message;}
});
ui.btnDiscardCorrection.addEventListener('click',()=>{state.developmentDraft=null;ui.developmentAssessment.value='';ui.developmentOutcome.value='';ui.developmentStatus.textContent='Browser-local correction draft discarded.';persistDevelopment();renderDevelopment();});
ui.btnExportCorrection.addEventListener('click',()=>{
  if(!state.developmentDraft?.coach_assessment)return;
  downloadBlob(new Blob([JSON.stringify(state.developmentDraft,null,2)],{type:'application/json'}),`touchline-coach-draft-${state.developmentDraft.id}.json`);
  ui.developmentStatus.textContent='Coach draft exported locally. It is not approved or shared.';
});
ui.btnSyncTime.addEventListener('click',()=>{ui.subStartInput.value=observationTime().toFixed(2);ui.subEndInput.value=(observationTime()+2.5).toFixed(2);});
function closeCue(){if(ui.subTextInput.value.trim()!==(state.editing?.text||'')&&!confirm('Discard unsaved cue changes?'))return;hideModal(ui.subtitleModal);}
ui.btnCancelSubModal.addEventListener('click',closeCue);
async function saveTranscript(next){
  const id=state.id;const result=await jsonResponse(await fetch(`${runUrl(id)}/transcript`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(next)}));
  if(result.status!=='ok'||!result.transcript)throw new Error('Transcript save was not confirmed');
  if(state.id===id&&state.data)state.data.transcript=result.transcript;
  renderCues();renderTimeline();renderFrame();
}
function cueLocation(cue){
  const transcript=state.data?.transcript;
  for(const key of ['segments','manual_cues']){
    const index=safeArray(transcript?.[key]).indexOf(cue);
    if(index>=0)return {key,index};
  }
  return null;
}
ui.btnSaveSubModal.addEventListener('click',async()=>{
  const start=Number(ui.subStartInput.value),end=Number(ui.subEndInput.value),text=ui.subTextInput.value.trim();
  if(!finite(start)||!finite(end)||start<0||end<=start||!text){ui.saveStatus.textContent='Enter text and a valid nonnegative time range.';return;}
  const transcript=structuredClone(state.data?.transcript||{schema_version:'touchline.transcript.v1',segments:[],manual_cues:[]});
  transcript.segments=safeArray(transcript.segments);transcript.manual_cues=safeArray(transcript.manual_cues);
  const cue=state.editing;
  if(cue){const location=cueLocation(cue),match=location&&transcript[location.key][location.index];if(!match){ui.saveStatus.textContent='Cue changed; reload the session before editing.';return;}Object.assign(match,{text,video_start_s:start,video_end_s:end});}
  else transcript.manual_cues.push({cue_id:`manual_${crypto.randomUUID()}`,text,video_start_s:start,video_end_s:end,language_code:ui.subLangSelect.value});
  ui.btnSaveSubModal.disabled=true;ui.saveStatus.textContent='Saving…';
  try{await saveTranscript(transcript);hideModal(ui.subtitleModal);toast('Cue saved.','success');}
  catch(error){ui.saveStatus.textContent=`Save failed: ${error.message}`;}
  finally{ui.btnSaveSubModal.disabled=false;}
});
async function deleteCue(cue){if(!confirm('Delete this cue from the saved transcript?'))return;
  const transcript=structuredClone(state.data?.transcript||{});
  const location=cueLocation(cue);if(!location){toast('Cue changed; reload before deleting.','error');return;}
  transcript[location.key].splice(location.index,1);
  try{await saveTranscript(transcript);toast('Cue deleted.','success');}catch(error){toast(`Delete failed: ${error.message}`,'error');}
}

function downloadBlob(blob,filename){
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
ui.exportDropdown.addEventListener('change',async event=>{
  const choice=event.target.value;event.target.value='';if(!choice)return;
  if(!state.id||!state.data){toast('Select a session first.','error');return;}
  if(choice==='clean_video'){
    if(!state.data.video_url){toast('Source video unavailable.','error');return;}
    const link=document.createElement('a');link.href=state.data.video_url;link.download=`touchline-source-${state.id}.mp4`;link.click();toast('Source-video download requested.','info');return;
  }
  if(choice==='srt'||choice==='vtt'){
    if(!allCues().length){toast('No cues to export.','error');return;}
    ui.exportDropdown.disabled=true;
    try{
      const response=await fetch(`${runUrl(state.id)}/subtitles.${choice}`);
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      downloadBlob(await response.blob(),`touchline-cues-${state.id}.${choice}`);
      toast(`${choice.toUpperCase()} export ready.`,'success');
    }catch(error){toast(`Export failed: ${error.message}`,'error');}
    finally{ui.exportDropdown.disabled=false;}
    return;
  }
  if(choice==='json'){downloadBlob(new Blob([JSON.stringify(state.data,null,2)],{type:'application/json'}),`touchline-loaded-${state.id}.json`);toast('Loaded session subset downloaded.','info');}
  if(choice==='markers'){downloadBlob(new Blob([JSON.stringify({schema_version:'touchline.local_markers.v1',session_id:state.id,markers:state.localMarkers},null,2)],{type:'application/json'}),`touchline-local-markers-${state.id}.json`);toast('Local marker export requested.','info');}
});
async function showUploadModal(){
  showModal(ui.uploadModal,state.demo?ui.uploadFileInput:ui.btnCancelUpload);
  if(state.demo)return;
  state.uploadReady=false;
  state.uploadEndpoint='/api/runs';
  ui.uploadFileInput.disabled=true;
  ui.uploadFramesSelect.disabled=true;
  ui.btnSubmitUpload.disabled=true;
  notice(ui.uploadStatusText,'Checking whether the authenticated full-recording job API is deployed…');
  try{
    const result=await api.readiness();
    if(!result.ready){notice(ui.uploadStatusText,result.message||'Video analysis is unavailable. Ask the operator to check the worker.');return;}
    state.uploadReady=true;
    state.uploadEndpoint=typeof result.upload_endpoint==='string'&&result.upload_endpoint.startsWith('/api/')?result.upload_endpoint:'/api/runs';
    ui.uploadFileInput.disabled=false;
    ui.uploadFramesSelect.disabled=false;
    ui.btnSubmitUpload.disabled=false;
    notice(ui.uploadStatusText,result.message||'Panorama ResNet50 worker is ready. Select an MP4 recording to start a backend analysis job.');
  }catch{
    notice(ui.uploadStatusText,'Video worker readiness could not be checked. Ask the operator before uploading.');
  }
}
function renderJob(){
  const job=state.job;
  ui.jobBanner.hidden=!job;
  if(!job){ui.btnUploadModal.disabled=false;if(state.view==='analysis')renderAnalysis();return;}
  const status=String(job.status||'processing').toLowerCase();
  ui.btnUploadModal.disabled=['uploading','queued','processing','unavailable'].includes(status);
  ui.jobBanner.dataset.status=status;
  ui.jobTitle.textContent=`${state.demo?'Synthetic demo · ':''}Analysis job ${job.id} · ${status.toUpperCase()}`;
  const messages={
    uploading:`Uploading ${job.fileName||'video'}${Number.isFinite(job.progress)?` · ${job.progress}%`:''}. No analysis has started yet.`,
    queued:'The backend accepted the job. Analysis has not started yet.',
    processing:'Upload accepted. The server is processing this job; results are not ready yet.',
    partial:'PARTIAL analysis is available. Frame coverage is limited; open the run to review what exists.',
    completed:'Analysis finished. Open the run to inspect its actual evidence.',
    failed:'Analysis failed. The upload may be retained on the server. Check the job status before retrying.',
    cancelled:'The job was cancelled. Check the upload before submitting another job.',
    unavailable:'Job status could not be checked. The upload may have succeeded; check status before retrying.'
  };
  const stage=typeof job.stage==='string'&&job.stage ? ` Current stage: ${job.stage}.` : '';
  const latest=Array.isArray(job.events)&&job.events.length ? job.events.at(-1) : null;
  const progress=latest&&typeof latest.event==='string' ? ` Latest event: ${latest.event}.` : '';
  ui.jobMessage.textContent=(messages[status]||`Server status: ${status}.`)+stage+progress+(state.demo?' The selected file was not analysed; this is a simulated response.':'');
  ui.jobProgress.hidden=status!=='uploading';
  if(status==='uploading')ui.jobProgress.value=job.progress||0;
  ui.btnJobOpen.hidden=!['partial','completed'].includes(status)||!state.runs.some(run=>run.id===job.id);
  ui.btnJobRetry.hidden=!['failed','unavailable','queued','processing','cancelled'].includes(status);
  ui.btnJobRetry.textContent=status==='failed'?'New upload':'Check status';
  ui.btnJobDismiss.hidden=['uploading','processing'].includes(status);
  if(state.view==='analysis')renderAnalysis();
}
function saveJob(){
  try{if(state.job?.id)sessionStorage.setItem('touchline.active_job.v1',state.job.id);else sessionStorage.removeItem('touchline.active_job.v1');}catch{}
}
async function pollJob(id){
  const token=++state.pollToken;
  for(let attempt=0;attempt<60&&!state.disposing&&token===state.pollToken;attempt++){
    try{
      const result=await api.getJob(id);
      if(token!==state.pollToken)return;
      const rawStatus=String(result.status||'unknown').toLowerCase();
      const status=rawStatus==='not_found'||rawStatus==='unknown'?'unavailable':rawStatus;
      state.job={...state.job,...result,id,status};renderJob();
      if(['partial','completed'].includes(status)){
        await loadSessions(id);
        if(state.id===id&&state.data)showView('workspace');
        return;
      }
      if(status==='failed')return;
      if(!['queued','processing'].includes(status))return;
    }catch(error){
      if(token!==state.pollToken)return;
      state.job={...state.job,id,status:'unavailable'};renderJob();return;
    }
    await new Promise(resolve=>setTimeout(resolve,2000));
  }
  if(token===state.pollToken&&state.job?.id===id){state.job.status='unavailable';renderJob();}
}
ui.btnJobOpen.addEventListener('click',()=>{if(state.job?.id)loadSession(state.job.id);});
ui.btnJobRetry.addEventListener('click',()=>{
  if(state.job?.status==='failed')showUploadModal();
  else if(state.job?.id)pollJob(state.job.id);
});
ui.btnJobDismiss.addEventListener('click',()=>{++state.pollToken;state.job=null;saveJob();renderJob();});
ui.btnUploadModal.addEventListener('click',showUploadModal);
ui.btnCancelUpload.addEventListener('click',()=>{if(state.job?.status!=='uploading')hideModal(ui.uploadModal);});
ui.uploadForm.addEventListener('submit',async event=>{
  event.preventDefault();const file=ui.uploadFileInput.files?.[0];if(!file)return;
  if(!state.demo&&!state.uploadReady){notice(ui.uploadStatusText,'Full-recording upload is unavailable until the authenticated backend job API is deployed. No file was sent.');return;}
  if(!/\.mp4$/i.test(file.name)||file.size<=0||file.size>1024*1024*1024){
    notice(ui.uploadStatusText,'Choose a non-empty MP4 no larger than 1 GiB. Larger authorized clips require the server upload limit to be configured.');return;
  }
  if(!state.demo&&!confirm('This upload starts a Panorama worker analysis job. Continue only if this footage and compute cost are approved.'))return;
  ui.btnSubmitUpload.disabled=true;
  notice(ui.uploadStatusText,state.demo?'Simulating upload; selected footage will not be analysed.':'Uploading footage…');
  state.job={id:null,status:'uploading',fileName:file.name,progress:0};renderJob();
  try{
    const query=new URLSearchParams({calibration:'uncalibrated',max_frames:ui.uploadFramesSelect.value});
    const endpoint=state.demo?`/api/analyze?${query}`:`${state.uploadEndpoint||'/api/runs'}?${query}`;
    const data=await TouchlineUpload.send(endpoint,file,progress=>{
      if(state.job?.status==='uploading'){state.job.progress=progress;renderJob();}
    });
    const jobId=data.job_id||data.run_id;
    state.job={id:jobId,status:data.status||'processing',stage:data.stage||'queued',fileName:file.name};saveJob();renderJob();
    hideModal(ui.uploadModal);ui.uploadFileInput.value='';
    pollJob(jobId);
  }catch(error){
    state.job={id:null,status:'failed',fileName:file.name};renderJob();
    notice(ui.uploadStatusText,error.message);
  }finally{ui.btnSubmitUpload.disabled=false;}
});
if(state.demo){
  ui.demoRibbon.hidden=false;
  ui.btnUploadModal.textContent='Simulate upload';
  const schoolCard=ui.overviewView.querySelector('.overview-boundaries .overview-empty');
  if(schoolCard&&window.__qaScenario==='two_sessions'){schoolCard.querySelector('strong').textContent='Demo Academy · fixture roster';schoolCard.querySelector('p').textContent='One fixture player label appears in two synthetic sessions. This is a demonstration association, not authenticated roster data or verified cross-session identity.';}
  ui.uploadModal.querySelector('.modal-help').textContent='SYNTHETIC DEMO: The selected MP4 is not uploaded or analysed. This fixture simulates an HTTP 202 job and a partial result. Use the real Studio server for authorized footage; real uploads may incur Modal costs.';
  ui.btnDemoStart.addEventListener('click',()=>loadSession('fixture-one'));
  ui.btnResetDemo.addEventListener('click',()=>{
    state.developmentDraft=null;persistDevelopment();
    for(const id of ['fixture-one','fixture-two'])try{localStorage.removeItem(markerStorageKey(id));}catch{}
    location.href='/web/qa/preview.html?scenario=two_sessions&view=overview';
  });
}else{
  try{const id=sessionStorage.getItem('touchline.active_job.v1');if(id){state.job={id,status:'processing'};renderJob();pollJob(id);}}catch{}
}
window.addEventListener('beforeunload',()=>{state.disposing=true;if(state.recorder?.state==='recording')state.recorder.stop();stopStream();discardAudio();});
clearSession('Loading available sessions…');loadSessions();refreshDevices();
