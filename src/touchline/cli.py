import argparse
import importlib.util
import json
from pathlib import Path
import platform
from .diagnostics import diagnostic_span, record_failure, DiagnosticUnavailable


class DiagnosticArgumentParser(argparse.ArgumentParser):
    def error(self,message):
        failure=record_failure(ValueError(),'cli')
        self.exit(2,f"error: invalid command arguments; use --help. Trace: {failure['trace_id']}\n")


def _main(argv=None):
    parser = DiagnosticArgumentParser(prog="touchline")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="CPU-only synthetic integration run; no model/field evidence")
    demo.add_argument("--output", type=Path, required=True)
    scan = sub.add_parser("inventory", help="Inspect local mixed data before choosing a detection-label adapter")
    scan.add_argument("root", type=Path)
    scan.add_argument("--output", type=Path, required=True)
    etl = sub.add_parser("etl", help="Explicit COCO/YOLO/corner/CSV box ETL; review and capture groups required")
    etl.add_argument("config", type=Path)
    etl.add_argument("--output", type=Path, required=True)
    soccer = sub.add_parser("stage-soccernet", help="Explicit reviewed MOT tracking subset to COCO staging")
    soccer.add_argument("config", type=Path)
    soccer.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate-detector")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--max-images", type=int)
    evaluate.add_argument("--allow-weight-download", action="store_true")
    evaluate.add_argument('--min-size',type=int)
    evaluate.add_argument('--max-size',type=int)
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--training-gate", action="store_true")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--vision", action="store_true")
    normalize = sub.add_parser('normalize-video', help='Explicit 24 FPS analysis copy at source resolution; video only, original unchanged')
    normalize.add_argument('video', type=Path)
    normalize.add_argument('--output', type=Path, required=True)
    preview = sub.add_parser('render-preview', help='CPU overlay video and contact sheet from saved observations; no inference')
    preview.add_argument('run_dir', type=Path)
    preview.add_argument('--output', type=Path, required=True)
    preview.add_argument('--source', type=Path)
    preview.add_argument('--max-seconds', type=float, default=20)
    preview.add_argument('--hold-seconds', type=float, default=.17)
    preview.add_argument('--focus-track-id', type=int)
    preview.add_argument('--show-proposals', action='store_true')
    retrack=sub.add_parser('retrack-run',help='ByteTrack replay of saved proposals; no video/model/GPU work')
    retrack.add_argument('run_dir',type=Path);retrack.add_argument('--output',type=Path,required=True)
    retrack.add_argument('--tracking-device',choices=['cpu','cuda'],default='cpu')
    retrack.add_argument('--activation-threshold',type=float,default=.5)
    retrack.add_argument('--max-gap-s',type=float,default=1.5)
    tracking_score=sub.add_parser('evaluate-tracking',help='IDF1 at IoU 0.5 on an explicitly reviewed sampled-frame reference')
    tracking_score.add_argument('run_dir',type=Path);tracking_score.add_argument('reference',type=Path)
    tracking_score.add_argument('--output',type=Path,required=True)
    media_check = sub.add_parser("analysis-preflight", help="CPU-only full-source landscape/24 CFR validation; no model/provider call")
    media_check.add_argument("video", type=Path)
    sub.add_parser("doctor", help="Local dependency/readiness diagnostics; never reads credentials or calls providers")
    diagnostic_report = sub.add_parser("diagnostic-report", help="Summarize one generated trace journal without forwarding raw log text")
    diagnostic_report.add_argument("trace",type=Path)
    export_check=sub.add_parser('export-check',help='Inspect known Colab export files/hashes without loading weights or credentials')
    export_check.add_argument('export_dir',type=Path)
    export_check.add_argument('--dataset-manifest',type=Path)
    export_check.add_argument('--verify-images',action='store_true')
    player=sub.add_parser('player-analysis',help='Local automatic evidence profiles; optional coach-rubric drafts')
    player.add_argument('run_dir',type=Path);player.add_argument('--output',type=Path,required=True)
    player.add_argument('--rubric',type=Path);player.add_argument('--session-context',type=Path);player.add_argument('--calibration',type=Path)
    prepare=sub.add_parser('repair-prepare',help='Offline allowlisted repair case; no model call or candidate execution')
    prepare.add_argument('source_root',type=Path);prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--failure-code',default='INTERNAL_ERROR')
    stage=sub.add_parser('repair-stage',help='Validate/stage a proposal without editing or executing live code')
    stage.add_argument('case',type=Path);stage.add_argument('proposal',type=Path)
    stage.add_argument('--source-root',type=Path,required=True);stage.add_argument('--output',type=Path,required=True)
    logs=sub.add_parser('run-log-export',help='Export run identity, diagnostic summary and numeric loss records locally')
    logs.add_argument('run_dir',type=Path);logs.add_argument('--output',type=Path,required=True);logs.add_argument('--diagnostics-dir',type=Path)
    analyze = sub.add_parser("analyze", help="Optional local PyAV + Torchvision baseline; no physical metrics")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--mode", choices=["fixed", "anchors24"], default="fixed")
    analyze.add_argument("--fps", type=float, default=6)
    analyze.add_argument("--checkpoint", type=Path)
    analyze.add_argument("--detector-endpoint", type=str, help="Optional protected image endpoint; Sandbox recording analysis remains primary")
    analyze.add_argument("--expected-checkpoint-sha256", help="Expected served checkpoint digest for the optional image endpoint")
    analyze.add_argument('--env-file',type=Path,help='OWNER ONLY: explicitly selected absolute settings file; never invoked by the agent')
    analyze.add_argument("--allow-weight-download", action="store_true")
    analyze.add_argument("--device", default=None)
    analyze.add_argument("--max-selected", type=int)
    analyze.add_argument("--min-size", type=int, default=1080, help="Detector resize short side; 1080 preserves the accepted source resolution")
    analyze.add_argument("--max-size", type=int, default=1920, help="Detector resize long side; default 1920 (benchmark VRAM before paid jobs)")
    analyze.add_argument("--record-proposals", action="store_true", help="Record raw detector proposals for stage-level audit (TL-030.1)")
    analyze.add_argument('--tracker',choices=['iou','bytetrack','bytetrack-cuda'],default='iou')
    analyze.add_argument("--resume", action="store_true", help="Resume an interrupted run in the output directory after verifying source and configuration")
    proposals = sub.add_parser("sam3-propose")
    proposals.add_argument("image", type=Path)
    proposals.add_argument("--checkpoint", type=Path, required=True)
    proposals.add_argument("--output", type=Path, required=True)
    proposals.add_argument("--device", default="cuda")
    train = sub.add_parser("train-detector")
    train.add_argument("manifest", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--max-steps", type=int, default=100)
    train.add_argument("--max-seconds", type=float, default=600)
    train.add_argument("--device", default="cuda")
    train.add_argument("--allow-weight-download", action="store_true")
    train.add_argument('--initial-checkpoint',type=Path,help='Warm-start selected weights with a fresh optimizer; not exact resume')
    train.add_argument('--learning-rate',type=float,default=.001)
    train.add_argument('--weight-decay',type=float,default=.0005)
    train.add_argument('--run-id');train.add_argument('--parent-run-id')
    train.add_argument('--execution-backend',choices=['python','colab','modal'],default='python',help='Provenance label only; does not launch a remote backend')
    audit_cmd = sub.add_parser("audit-run", help="Produce a stage-by-stage miss ledger for a run against ground-truth boxes")
    audit_cmd.add_argument("run_dir", type=Path, help="Directory containing run artifacts (manifest.json, coverage.jsonl, observations.jsonl)")
    audit_cmd.add_argument("ground_truth", type=Path, help="Path to JSON file containing ground-truth annotations")
    audit_cmd.add_argument("--output", type=Path, required=True, help="Destination path for audit ledger JSON")
    audit_cmd.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for detection matching (default: 0.5)")
    inspect_cmd = sub.add_parser("inspect", help="Generate player analytics and interactive dashboard for a run directory")
    inspect_cmd.add_argument("run_dir", type=Path, help="Directory containing run artifacts (manifest.json, observations.jsonl)")
    inspect_cmd.add_argument("--output", type=Path, help="Custom output path for analytics.json (defaults to <run_dir>/analytics.json)")
    inspect_cmd.add_argument("--dashboard", type=Path, help="Custom output path for dashboard HTML (defaults to <run_dir>/review.html)")
    inspect_cmd.add_argument("--calibration", type=Path, default=None, help="Optional path to pitch calibration JSON")
    associate_cmd = sub.add_parser("associate", help="Explicitly associate a track with a confirmed player identity (TL-004)")
    associate_cmd.add_argument("run_dir", type=Path, help="Directory containing run artifacts (manifest.json, observations.jsonl)")
    associate_cmd.add_argument("--track", type=int, required=True, help="Local track ID to associate")
    associate_cmd.add_argument("--player", type=str, required=True, help="Confirmed player ID (e.g. P07, Player_Smith)")
    associate_cmd.add_argument("--name", type=str, help="Optional human-readable player name")
    associate_cmd.add_argument("--target", action="store_true", help="Mark as target participant")
    associate_cmd.add_argument("--reviewer", type=str, default="coach_manual_review", help="Reviewer identity")
    associate_cmd.add_argument("--notes", type=str, help="Review notes or justification for association/correction")
    studio_cmd = sub.add_parser("studio", help="Launch local Touchline Live Studio web UI for MP4 upload, calibration, and speed dashboard")
    studio_cmd.add_argument("--runs-dir", type=Path, default=None, help="Runs directory; configured path or ./runs by default")
    studio_cmd.add_argument('--env-file',type=Path,help='OWNER ONLY: explicitly selected absolute settings file')
    studio_cmd.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")

    args = parser.parse_args(argv)
    try:
        if args.command=='player-analysis':
            from .player_analysis import analyze_run
            from .metrics import Calibration
            def optional_json(path):
                if path is None:return None
                if path.suffix.lower()!='.json':raise ValueError('Explicit JSON input required, never a credential file')
                return json.loads(path.read_text(encoding='utf-8'))
            calibration=optional_json(args.calibration)
            analysis=analyze_run(args.run_dir,rubric=optional_json(args.rubric),session_context=optional_json(args.session_context),
                calibration=Calibration(**calibration) if calibration is not None else None)
            args.output.mkdir(parents=True,exist_ok=False)
            (args.output/'player-analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            rows=['# Automatic player evidence profiles','','Draft only; coach review and independent quality acceptance remain required.','']
            for profile in analysis['profiles']:
                rows+=['## Track '+str(profile['local_track_id']),'','Identity: '+str(profile['player_id'] or 'unresolved'),'']
                rows+=['- '+name+': '+str(value['value'])+' '+value['units']+'; '+str(value['reason'] or 'supported estimate') for name,value in profile['metrics'].items()]
                rows+=['']
            (args.output/'review.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
            print(json.dumps({'run_id':analysis['lineage']['run_id'],'profiles':len(analysis['profiles']),
                'unresolved_tracks':len(analysis['unresolved_tracks']),'approval_state':analysis['approval_state']}))
            return 0
        if args.command=='repair-prepare':
            from .repair import prepare_case
            case=prepare_case(args.source_root,failure_code=args.failure_code)
            if args.output.exists():raise ValueError('Use a new repair case file')
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(case,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'case_id':case['case_id'],'baseline_id':case['baseline_id'],'execution':'not performed'}))
            return 0
        if args.command=='repair-stage':
            from .repair import stage_proposal
            if any(p.suffix.lower()!='.json' for p in (args.case,args.proposal)):raise ValueError('Explicit case/proposal JSON required')
            packet=json.loads(args.case.read_text(encoding='utf-8'));proposal=json.loads(args.proposal.read_text(encoding='utf-8'))
            result=stage_proposal(packet,proposal,args.output,source_root=args.source_root)
            print(json.dumps({'candidate_id':result['candidate_id'],'promotion_state':result['promotion_state'],'live_source_changed':False}))
            return 0
        if args.command=='run-log-export':
            from .run_records import export_run_logs
            print(json.dumps(export_run_logs(args.run_dir,args.output,diagnostics_dir=args.diagnostics_dir),indent=2))
            return 0
        if args.command=='export-check':
            from .artifacts import inspect_colab_export
            print(json.dumps(inspect_colab_export(args.export_dir,dataset_manifest=args.dataset_manifest,verify_images=args.verify_images),indent=2))
            return 0
        if args.command == "doctor":
            from .diagnostics import local_doctor
            print(json.dumps(local_doctor(), indent=2))
            return 0
        if args.command == "diagnostic-report":
            from .diagnostics import diagnostic_summary
            print(json.dumps(diagnostic_summary(args.trace), indent=2))
            return 0
        if args.command == 'retrack-run':
            from .tracking_replay import retrack_run
            print(json.dumps(retrack_run(args.run_dir,args.output,prediction_device=args.tracking_device,
                activation_threshold=args.activation_threshold,max_gap_s=args.max_gap_s),indent=2))
            return 0
        if args.command == 'evaluate-tracking':
            from .tracking_evaluation import evaluate_tracking
            report=evaluate_tracking(args.run_dir,args.reference)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8') as handle:
                handle.write(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
            return 0
        if args.command == 'render-preview':
            from .video.preview import render_run_preview
            print(json.dumps(render_run_preview(args.run_dir, args.output, source=args.source,
                max_seconds=args.max_seconds, hold_seconds=args.hold_seconds,
                focus_track_id=args.focus_track_id, show_proposals=args.show_proposals), indent=2))
            return 0
        if args.command == 'normalize-video':
            from .video.normalize import normalize_video
            print(normalize_video(args.video, args.output))
            return 0
        if args.command == "analysis-preflight":
            from .analysis_policy import preflight_analysis_video
            print(json.dumps(preflight_analysis_video(args.video), indent=2))
            return 0
        if args.command == "stage-soccernet":
            from .data.soccernet import stage_soccernet
            print(stage_soccernet(args.config, args.output))
            return 0
        if args.command == "evaluate-detector":
            from .training.evaluate import evaluate_detector
            print(json.dumps(evaluate_detector(args.manifest, args.output, args.checkpoint, args.split, args.device, args.max_images, args.allow_weight_download,
                min_size=args.min_size,max_size=args.max_size), indent=2))
            return 0
        if args.command == "inventory":
            from .data.inventory import inventory
            if args.output.exists():
                raise ValueError("output already exists")
            result = inventory(args.root)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print("Saved inventory; data is not yet training-ready.")
            return 0
        if args.command == "etl":
            from .data.etl import build_release
            data = build_release(args.config, args.output)
            print(json.dumps({"release_id": data["release_id"], "samples": len(data["samples"]), "fixture_only": data["fixture_only"]}, indent=2))
            return 0
        if args.command == "preflight":
            result = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "modules": {name: importlib.util.find_spec(name) is not None for name in ["torch", "torchvision", "av", "PIL", "sam3"]},
            }
            if args.vision:
                import torch, torchvision
                result.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_available=torch.cuda.is_available())
                if torch.cuda.is_available():
                    result["gpu"] = {"name": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory}
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "validate-dataset":
            from .data.manifest import read_manifest
            data = read_manifest(args.manifest, args.training_gate)
            print(json.dumps({
                "status": "metadata_passed",
                "release_id": data["release_id"],
                "fixture_only": data.get("fixture_only", True),
                "image_files_verified": args.training_gate,
                "not_checked": "label correctness, unknown near duplicates, mask geometry, model/metric validity",
            }, indent=2))
            return 0
        if args.command in ("demo", "analyze"):
            from .pipeline import run_pipeline
            from .video.sampling import FixedRateSampler, Anchor24Sampler
            if args.command == "demo":
                from .fixtures import synthetic_frames, SyntheticDetector
                result = run_pipeline(synthetic_frames(), SyntheticDetector(), Anchor24Sampler(), args.output, {"id": "synthetic-source", "not_real_video": True}, synthetic=True)
            else:
                settings=None
                if args.env_file is not None:
                    from .settings import load_operator_settings
                    settings=load_operator_settings(args.env_file)
                checkpoint=args.checkpoint or (settings.checkpoint_path if settings is not None else None)
                device=args.device or (settings.device if settings is not None else 'cpu')
                if checkpoint is not None and (checkpoint.suffix.lower() not in ('.pt','.pth') or not checkpoint.is_file()):
                    raise ValueError('Select existing trained .pt/.pth weights, not the dataset release.json')
                from .adapters.torchvision import TorchvisionDetector
                from .data.manifest import sha256_file
                from .video.decode import decode_video
                if not args.video.is_file():
                    raise ValueError("video does not exist")
                from .analysis_policy import preflight_analysis_video
                tracker=None
                if args.tracker in ('bytetrack','bytetrack-cuda'):
                    from .byte_tracking import ByteTrackTracker
                    tracker=ByteTrackTracker(2 if args.mode=='anchors24' else args.fps,
                        prediction_device='cuda' if args.tracker=='bytetrack-cuda' else 'cpu')
                    if args.resume:raise ValueError('ByteTrack resume requires saved motion state; start a fresh run')
                media_policy = preflight_analysis_video(args.video)
                if getattr(args, "detector_endpoint", None):
                    from .detector_api import DetectorApiClient
                    token_id = settings.credentials.get("MODAL_PROXY_TOKEN_ID") if settings is not None else None
                    token_sec = settings.credentials.get("MODAL_PROXY_TOKEN_SECRET") if settings is not None else None
                    detector = DetectorApiClient(
                        args.detector_endpoint,
                        token_id=token_id,
                        token_secret=token_sec,
                        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                    )
                else:
                    detector = TorchvisionDetector(checkpoint, device, threshold=.1 if tracker is not None else .5,
                        allow_download=args.allow_weight_download, min_size=args.min_size, max_size=args.max_size)
                fingerprint = sha256_file(args.video)
                sampler = Anchor24Sampler() if args.mode == "anchors24" else FixedRateSampler(args.fps)
                result = run_pipeline(
                    decode_video(args.video, fingerprint),
                    detector,
                    sampler,
                    args.output,
                    {"id": fingerprint, "path": str(args.video.resolve()), "stream_index": 0, "analysis_policy": media_policy},
                    max_selected=args.max_selected,
                    resume=args.resume,
                    record_proposals=args.record_proposals,
                    tracker=tracker,
                )
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "sam3-propose":
            from .adapters.sam3 import propose_people
            if args.output.exists():
                raise ValueError("output already exists")
            result = propose_people(args.image, args.checkpoint, args.device)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print("Saved unreviewed proposals; not a training release.")
            return 0
        if args.command == "train-detector":
            from .training.runner import train_detector
            result = train_detector(args.manifest, args.output, args.epochs, args.max_steps, args.max_seconds, args.device, args.allow_weight_download,
                run_id=args.run_id,execution_backend=args.execution_backend,parent_run_id=args.parent_run_id,initial_checkpoint=args.initial_checkpoint,
                learning_rate=args.learning_rate,weight_decay=args.weight_decay)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "audit-run":
            from .audit import audit_ground_truth
            if not args.run_dir.is_dir():
                raise ValueError(f"run_dir does not exist or is not a directory: {args.run_dir}")
            if not args.ground_truth.is_file():
                raise ValueError(f"ground_truth file does not exist: {args.ground_truth}")
            if args.output.resolve() == args.ground_truth.resolve():
                raise ValueError("output path cannot be the same as ground_truth input path")
            try:
                gt_data = json.loads(args.ground_truth.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid JSON in ground_truth file: {exc}") from exc
            ledger = audit_ground_truth(args.run_dir, gt_data, iou_threshold=args.iou_threshold)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"status": "audit_complete", "run_id": ledger.get("run_id"), "summary": ledger.get("summary"), "output": str(args.output)}, indent=2))
            return 0
        if args.command == "inspect":
            from .analytics import generate_match_analytics
            from .viewer import render_inspection_dashboard
            if not args.run_dir.is_dir():
                raise ValueError(f"run_dir does not exist or is not a directory: {args.run_dir}")
            manifest_file = args.run_dir / "manifest.json"
            obs_file = args.run_dir / "observations.jsonl"
            if not manifest_file.is_file():
                raise ValueError(f"manifest.json missing in run_dir: {args.run_dir}")
            if not obs_file.is_file():
                raise ValueError(f"observations.jsonl missing in run_dir: {args.run_dir}")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            observations = []
            with obs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        observations.append(json.loads(line))

            assoc_file = args.run_dir / "associations.json"
            associations = None
            if assoc_file.is_file():
                try:
                    associations = json.loads(assoc_file.read_text(encoding="utf-8"))
                except Exception:
                    associations = None

            from .metrics import Calibration
            cal = None
            cal_path = getattr(args, "calibration", None) or (args.run_dir / "calibration.json" if (args.run_dir / "calibration.json").is_file() else None)
            if cal_path is not None and cal_path.is_file():
                try:
                    cal = Calibration(**json.loads(cal_path.read_text(encoding="utf-8")))
                except Exception:
                    cal = None

            tfile = args.run_dir / "transcript.json"
            transcript = None
            if tfile.is_file():
                try:
                    transcript = json.loads(tfile.read_text(encoding="utf-8"))
                except Exception:
                    transcript = None

            analytics = generate_match_analytics(manifest, observations, calibration=cal, associations=associations)
            out_json = args.output or (args.run_dir / "analytics.json")
            out_dash = args.dashboard or (args.run_dir / "review.html")
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_dash.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")
            out_dash.write_text(render_inspection_dashboard(manifest, analytics, observations, transcript=transcript), encoding="utf-8")
            summary = {
                "status": "inspection_ready",
                "run_id": analytics["run_id"],
                "synthetic": analytics["synthetic"],
                "total_tracks": analytics["quality_indicators"]["total_tracks"],
                "confirmed_players_count": analytics["quality_indicators"].get("confirmed_players_count", 0),
                "target_tracks_count": analytics["quality_indicators"].get("target_tracks_count", 0),
                "observation_count": analytics["session"]["observation_count"],
                "calibrated": analytics["calibration"]["calibrated"],
                "analytics_json": str(out_json),
                "dashboard_html": str(out_dash),
            }
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "associate":
            from .tracking import PlayerAssociationLedger
            from .analytics import generate_match_analytics
            from .viewer import render_inspection_dashboard
            if not args.run_dir.is_dir():
                raise ValueError(f"run_dir does not exist or is not a directory: {args.run_dir}")
            manifest_file = args.run_dir / "manifest.json"
            obs_file = args.run_dir / "observations.jsonl"
            if not manifest_file.is_file():
                raise ValueError(f"manifest.json missing in run_dir: {args.run_dir}")
            if not obs_file.is_file():
                raise ValueError(f"observations.jsonl missing in run_dir: {args.run_dir}")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            run_id = manifest.get("run_id", "unknown_run")

            assoc_file = args.run_dir / "associations.json"
            if assoc_file.is_file():
                try:
                    ledger = PlayerAssociationLedger.from_dict(json.loads(assoc_file.read_text(encoding="utf-8")))
                    if ledger.run_id != run_id:
                        raise ValueError(f"association ledger run_id mismatch: {ledger.run_id} vs {run_id}")
                except Exception as exc:
                    if "mismatch" in str(exc):
                        raise
                    ledger = PlayerAssociationLedger(run_id=run_id)
            else:
                ledger = PlayerAssociationLedger(run_id=run_id)

            assoc = ledger.associate(
                local_track_id=args.track,
                player_id=args.player,
                player_name=args.name,
                is_target=args.target,
                reviewed_by=args.reviewer,
                notes=args.notes,
            )
            assoc_file.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")

            # Re-generate analytics and update dashboard
            observations = []
            with obs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        observations.append(json.loads(line))
            cal_file = args.run_dir / "calibration.json"
            cal = None
            if cal_file.is_file():
                try:
                    from .metrics import Calibration
                    cal = Calibration(**json.loads(cal_file.read_text(encoding="utf-8")))
                except Exception:
                    cal = None
            tfile = args.run_dir / "transcript.json"
            transcript = None
            if tfile.is_file():
                try:
                    transcript = json.loads(tfile.read_text(encoding="utf-8"))
                except Exception:
                    transcript = None
            analytics = generate_match_analytics(manifest, observations, calibration=cal, associations=ledger)
            (args.run_dir / "analytics.json").write_text(json.dumps(analytics, indent=2) + "\n", encoding="utf-8")
            (args.run_dir / "review.html").write_text(render_inspection_dashboard(manifest, analytics, observations, transcript=transcript), encoding="utf-8")

            summary = {
                "status": "association_recorded",
                "run_id": run_id,
                "local_track_id": assoc.local_track_id,
                "player_id": assoc.player_id,
                "player_name": assoc.player_name,
                "is_target": assoc.is_target,
                "revision": assoc.revision,
                "corrections_count": len(ledger.corrections),
                "ledger_file": str(assoc_file),
            }
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "studio":
            from .studio import run_studio_server
            settings=None
            if args.env_file is not None:
                from .settings import load_operator_settings
                settings=load_operator_settings(args.env_file)
            runs=args.runs_dir or (settings.runs_dir if settings is not None else Path('runs'))
            run_studio_server(runs, port=args.port, settings=settings)
            return 0
    except Exception as exc:
        failure=record_failure(exc,'cli')
        parser.exit(2, f"error: {failure['code']}: {failure['message']} Trace: {failure['trace_id']}\n")
    return 0


def main(argv=None):
    try:
        with diagnostic_span('cli'):
            return _main(argv)
    except DiagnosticUnavailable:
        raise SystemExit('Diagnostics unavailable; stop and check the private log directory before continuing.') from None
