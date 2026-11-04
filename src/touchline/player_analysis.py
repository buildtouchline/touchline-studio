"""Automatic evidence profiles and deterministic, explicitly provisional rubric drafts."""
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from .analytics import generate_match_analytics
from .contracts import deserialize_observation
from .diagnostics import diagnosed
from .tracking import reviewed_association

METRICS={'distance_m':'metres','average_speed_mps':'metres_per_second',
         'maximum_windowed_speed_mps':'metres_per_second'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _number(value):return type(value) in (int,float) and math.isfinite(value)


@diagnosed('player.profiles')
def generate_profiles(manifest,observations,*,associations=None,calibration=None,session_context=None,max_observations=200000,max_gap_s=0.75):
    if not isinstance(manifest,dict) or not isinstance(manifest.get('run_id'),str) or not manifest['run_id']:
        raise ValueError('A source run ID is required')
    if not _number(max_gap_s) or max_gap_s<=0:raise ValueError('Positive gap limit required')
    if type(max_observations) is not int or max_observations<1:raise ValueError('Positive observation limit required')
    selected=manifest.get('selected_frames');decoded=manifest.get('decoded_frames')
    if type(selected) is not int or type(decoded) is not int or not 0<=selected<=decoded:
        raise ValueError('Recorded decoded/selected coverage counts are required')
    records=[];groups={};seen=set();origin=None;times={};source_ids=set()
    for raw in observations:
        if len(records)>=max_observations:raise ValueError('Observation limit exceeded; split the reviewed run explicitly')
        parsed=deserialize_observation(raw)
        if type(parsed.local_track_id) is not int:raise ValueError('Exact integer local track ID required')
        if parsed.run_id!=manifest['run_id']:raise ValueError('Cross-run observation rejected')
        if raw.get('synthetic') is True and not manifest.get('synthetic'):raise ValueError('Synthetic observations cannot become real evidence')
        if not parsed.box.fits(parsed.frame.width,parsed.frame.height):raise ValueError('Observation box exceeds source geometry')
        frame=parsed.frame;key=(parsed.local_track_id,frame.index)
        if key in seen:raise ValueError('Duplicate track/frame observation')
        seen.add(key);source_ids.add(frame.source_id)
        if len(source_ids)>1:raise ValueError('Mixed video sources require separate runs')
        previous=times.get(parsed.local_track_id)
        if previous is not None and frame.timestamp_s<=previous:raise ValueError('Track timestamps must increase')
        times[parsed.local_track_id]=frame.timestamp_s
        offset=Fraction(frame.pts*frame.time_base_num,frame.time_base_den)-Fraction(str(frame.timestamp_s))
        if origin is None:origin=offset
        if abs(offset-origin)>Fraction(1,1000000):raise ValueError('Observation PTS and relative time disagree')
        records.append(raw)
        groups.setdefault(parsed.local_track_id,[]).append(raw)
    if any(len(items)>selected for items in groups.values()):
        raise ValueError('Track support exceeds recorded selected-frame coverage')
    expected=manifest.get('source',{}).get('id') if isinstance(manifest.get('source'),dict) else None
    if records and not manifest.get('synthetic') and (not expected or source_ids!={expected}):
        raise ValueError('Observation source does not match the run manifest')
    analytics=generate_match_analytics(manifest,records,calibration=calibration,associations=associations,max_gap_s=max_gap_s)
    lineage={'run_id':manifest['run_id'],'source':manifest.get('source'),
        'manifest_sha256':digest(manifest),'observations_sha256':digest(records),
        'detector':manifest.get('detector'),'sampler':manifest.get('sampler'),
        'sampler_config':manifest.get('sampler_config'),'calibration':asdict(calibration) if calibration else None,
        'session_context':session_context}
    profiles=[]
    for track in analytics['player_roster']:
        tid=track['local_track_id'];items=groups[tid]
        review=reviewed_association(track.get('identity_review'),run_id=manifest['run_id'],local_track_id=tid)
        evidence={'frame_indices':[r['frame']['index'] for r in items],
            'pts':[r['frame']['pts'] for r in items],
            'time_bases':[[r['frame']['time_base_num'],r['frame']['time_base_den']] for r in items],
            'start_s':track['start_time_s'],'end_s':track['end_time_s'],
            'observation_count':len(items),'source_observations_sha256':lineage['observations_sha256']}
        support=len(items)/selected if selected else None
        metrics={}
        for name,units in METRICS.items():
            value=track['movement'].get(name)
            eligible=review is not None and track['movement'].get('calibrated') is True and _number(value)
            reason=None if eligible else ('identity_not_reviewed' if review is None else 'calibration_or_motion_support_unavailable')
            metrics[name]={'value':value if eligible else None,'units':units,'eligible':eligible,
                'reason':reason,'method':'reviewed_ground_contact_projection_v1','evidence':evidence,
                'eligible_duration_s':track['movement'].get('eligible_duration_s'),
                'excluded_segments':track['movement'].get('excluded_segments',[])}
        profile={'schema':'touchline.player-profile.v1','run_id':manifest['run_id'],
            'local_track_id':tid,'player_id':review['player_id'] if review else None,
            'identity_status':'locally_reviewed' if review else 'unresolved','identity_review':review,
            'synthetic':bool(manifest.get('synthetic',False)),'lineage':lineage,'evidence':evidence,
            'sample_support':{'ratio':support,'observed_frames':len(items),'denominator_selected_frames':selected,
                'decoded_frames':decoded,'selected_fraction':selected/decoded if decoded else None,
                'remaining_source_coverage':'unknown' if manifest.get('status')!='complete' else 'recorded_source_only',
                'interpretation':'support on selected frames; not continuous visibility or player ability'},
            'metrics':metrics,'activity_classification':track.get('activity_classification'),
            'intensity_zones':track.get('intensity_zones'),
            'unsupported':{'passes':'no_validated_event_extractor','shots':'no_validated_event_extractor',
                'tackles':'no_validated_event_extractor','fitness':'not_inferred'},
            'status':'automatic_draft','approved':False}
        profile['profile_id']=digest(profile);profiles.append(profile)
    return {'schema':'touchline.player-analysis.v1','lineage':lineage,'profiles':profiles,
        'unresolved_tracks':[p['local_track_id'] for p in profiles if p['player_id'] is None],
        'coverage':{'observations':len(records),'observed_tracks':len(profiles),'selected_frames':selected,'decoded_frames':decoded},
        'quality_acceptance':'unverified; fixture success and detector confidence are not 95-percent accuracy',
        'approval_state':'draft_pending_coach_review'}


def validate_rubric(rubric):
    if not isinstance(rubric,dict) or rubric.get('schema')!='touchline.rubric.v1':raise ValueError('Unsupported rubric')
    for field in ('rubric_id','version','drill_id','context'):
        if not isinstance(rubric.get(field),str) or not rubric[field].strip():raise ValueError('Rubric identity/context is required')
    dimensions=rubric.get('dimensions')
    if not isinstance(dimensions,list) or not dimensions:raise ValueError('Declare rubric dimensions explicitly')
    seen=set();weight=0
    for d in dimensions:
        if not isinstance(d,dict) or d.get('metric') not in METRICS or d['metric'] in seen:
            raise ValueError('Unsupported or repeated rubric metric')
        seen.add(d['metric'])
        if d.get('units')!=METRICS[d['metric']]:raise ValueError('Rubric units disagree')
        if not all(_number(d.get(k)) for k in ('lower','upper','weight','minimum_support_ratio','focus_below')):
            raise ValueError('Finite rubric parameters required')
        if not d['lower']<d['upper'] or d['weight']<=0 or not 0<=d['minimum_support_ratio']<=1 or not 0<=d['focus_below']<=100:
            raise ValueError('Invalid rubric bounds/weight/support')
        if d.get('direction') not in ('higher','lower') or type(d.get('minimum_observations')) is not int or d['minimum_observations']<2:
            raise ValueError('Explicit direction and observation minimum required')
        template=d.get('plan_template')
        if not isinstance(template,dict) or any(not isinstance(template.get(k),str) or not template[k].strip() for k in ('drill','success_criterion','review_window')):
            raise ValueError('Coach drill/criterion/review template required')
        weight+=d['weight']
    if not math.isclose(weight,1.0,abs_tol=1e-9):raise ValueError('Declared weights must sum to one; never silently renormalize')


@diagnosed('player.score')
def score_profiles(analysis,rubric):
    validate_rubric(rubric);rubric_hash=digest(rubric);results=[]
    reviewed=rubric.get('reviewed') is True and isinstance(rubric.get('reviewed_by'),str) and bool(rubric['reviewed_by'].strip())
    for profile in analysis['profiles']:
        if profile.get('profile_id')!=digest({k:v for k,v in profile.items() if k!='profile_id'}):
            raise ValueError('Profile changed; regenerate evidence before scoring')
        context=profile['lineage'].get('session_context')
        compatible=isinstance(context,dict) and context.get('reviewed') is True and isinstance(context.get('reviewed_by'),str) and bool(context['reviewed_by'].strip()) and all(context.get(k)==rubric[k] for k in ('drill_id','context'))
        dimensions=[];plan=[]
        for rule in rubric['dimensions']:
            metric=profile['metrics'][rule['metric']];support=profile['sample_support']['ratio']
            reason=None
            if not reviewed:reason='rubric_not_locally_reviewed'
            elif profile['synthetic']:reason='synthetic_fixture_not_player_evidence'
            elif not compatible:reason='drill_context_not_reviewed_or_incompatible'
            elif not metric['eligible']:reason=metric['reason']
            elif support is None or support<rule['minimum_support_ratio'] or profile['evidence']['observation_count']<rule['minimum_observations']:
                reason='insufficient_observed_support'
            points=None
            if reason is None:
                fraction=(metric['value']-rule['lower'])/(rule['upper']-rule['lower'])
                if rule['direction']=='lower':fraction=1-fraction
                points=round(100*min(1,max(0,fraction)),4)
                if points<rule['focus_below']:
                    plan.append({'metric':rule['metric'],'points':points,'evidence':metric['evidence'],
                        'template':rule['plan_template'],'status':'draft_requires_coach_review'})
            dimensions.append({'metric':rule['metric'],'raw_value':metric['value'],'units':metric['units'],
                'points':points,'weight':rule['weight'],'reason':reason})
        total=round(sum(d['points']*d['weight'] for d in dimensions),4) if all(d['points'] is not None for d in dimensions) else None
        results.append({'profile_id':profile['profile_id'],'player_id':profile['player_id'],'local_track_id':profile['local_track_id'],
            'rubric_sha256':rubric_hash,'dimensions':dimensions,'coach_rubric_total':total,'improvement_plan_draft':plan,
            'approval_state':'draft_pending_coach_review','progress_claim':'one-session baseline; no demonstrated improvement'})
    return {'schema':'touchline.player-score-drafts.v1','rubric_sha256':rubric_hash,'results':results,'approved':False}


@diagnosed('player.profiles')
def analyze_run(run_dir,*,rubric=None,calibration=None,session_context=None):
    original=Path(run_dir).absolute()
    if any(p.is_symlink() for p in (original,*original.parents)):raise ValueError('Unsafe run directory')
    root=original.resolve()
    def path(name):
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('Unsafe run artifact')
        return p
    manifest=json.loads(path('manifest.json').read_text(encoding='utf-8'))
    associations=json.loads(path('associations.json').read_text(encoding='utf-8')) if path('associations.json').is_file() else None
    with path('observations.jsonl').open(encoding='utf-8') as stream:
        analysis=generate_profiles(manifest,(json.loads(line) for line in stream if line.strip()),associations=associations,calibration=calibration,session_context=session_context)
    if rubric is not None:analysis['score_drafts']=score_profiles(analysis,rubric)
    return analysis
