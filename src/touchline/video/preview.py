"""Bounded visual review from saved observations; never runs a detector or provider."""
from fractions import Fraction
import json
import math
from pathlib import Path

from touchline.data.manifest import sha256_file
from touchline.video.decode import decode_video, probe_video


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _track_color(track_id):
    return ('#00ff80','#00d4ff','#ff66ff','#ffe066','#b0a0ff','#ff884d')[track_id % 6]


def _badge(draw, box, text, color, font, occupied, width, height, top):
    """Try several label positions, with a leader line back to the player."""
    bounds=draw.textbbox((0,0),text,font=font)
    w=min(width,bounds[2]-bounds[0]+16);h=bounds[3]-bounds[1]+12
    anchor=(box[0]+box[2])/2
    left=max(0,min(width-w,int(anchor-w/2)))
    base=max(top,min(height-h,int(box[1]-h-6)))
    candidate=(left,base,left+w,base+h)
    for dx,dy in ((0,0),(0,-h-4),(w+4,0),(-w-4,0),(0,-2*h-8),(w+4,-h-4),(-w-4,-h-4)):
        x=max(0,min(width-w,left+dx));y=max(top,min(height-h,base+dy))
        trial=(x,y,x+w,y+h)
        if not any(trial[0]<r[2] and trial[2]>r[0] and trial[1]<r[3] and trial[3]>r[1] for r in occupied):
            candidate=trial;break
    occupied.append(candidate)
    x,y,x2,y2=candidate
    draw.line(((x+x2)/2,y2,anchor,box[1]),fill=color,width=2)
    draw.rectangle(candidate,fill='black',outline=color,width=2)
    draw.text((x+8,y+6-bounds[1]),text,font=font,fill='white')


def render_run_preview(run_dir, output, *, source=None, max_seconds=20, hold_seconds=0.17,
                       focus_track_id=None, show_proposals=False, full_run=False):
    """Write a source-resolution 24 FPS overlay, contact sheet and provenance.

    Large colored labels identify temporary tracks, never jerseys or names.
    An optional focused crop follows one recorded track, without filling ID gaps.
    Amber proposals are opt-in. Held overlays are labelled, never new evidence.
    Existing output directories are refused. No observations or run files change.
    """
    import av
    from PIL import Image, ImageDraw
    run_dir, output = Path(run_dir), Path(output)
    if focus_track_id is not None and (type(focus_track_id) is not int or focus_track_id < 0):
        raise ValueError('Focus track must be an exact nonnegative integer or None')
    if type(show_proposals) is not bool:
        raise ValueError('show_proposals must be boolean')
    if type(full_run) is not bool or full_run and max_seconds is not None:
        raise ValueError('Full analyzed-run rendering requires full_run=True and max_seconds=None')
    if not full_run and (not isinstance(max_seconds, (int,float)) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 120):
        raise ValueError('Preview duration must be positive and at most 120 seconds')
    if not isinstance(hold_seconds, (int,float)) or not math.isfinite(hold_seconds) or not 0 <= hold_seconds <= .5:
        raise ValueError('Visual hold must be between zero and 0.5 seconds')
    manifest = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    source = Path(source) if source is not None else Path(manifest['source']['path'])
    original_hash = sha256_file(source)
    if original_hash != manifest['source']['id']:
        raise ValueError('Preview source hash differs from the analyzed video')
    info = probe_video(source)
    if info['average_fps'] != 24 or info['is_rotated']:
        raise ValueError('Preview requires the same normalized 24 FPS analysis input')
    limit = manifest['decoded_frames'] if full_run else min(math.ceil(max_seconds*24), manifest['decoded_frames'])
    if limit <= 0:
        raise ValueError('No processed video frames to preview')
    coverage, tracks, proposals = {}, {}, {}
    for name, target in (('coverage.jsonl',coverage),('observations.jsonl',tracks),('raw_proposals.jsonl',proposals)):
        path = run_dir/name
        if not path.is_file():
            continue
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                item = json.loads(line)
                index = item['frame']['index']
                if index >= limit:
                    break
                if name == 'coverage.jsonl':
                    coverage[index] = item['state']
                else:
                    target.setdefault(index, []).append(item)
    sampled = sorted(i for i,state in coverage.items() if state != 'unsampled_by_sampler')
    if not sampled:
        raise ValueError('No sampled coverage in the selected preview interval')
    track_index={}
    for index,rows in tracks.items():
        for row in rows:
            tid=row['local_track_id']
            entry=track_index.setdefault(tid,{'track_id':tid,'label':f'T{tid}','first_frame':index,'last_frame':index,'observations':0})
            entry['last_frame']=index;entry['observations']+=1
    if focus_track_id is not None and focus_track_id not in track_index:
        raise ValueError('Selected track is not observed in this preview interval; consult track-index.json or choose another interval')
    sheet_indices = {sampled[round(i*(len(sampled)-1)/5)] for i in range(6)}
    output.mkdir(parents=True, exist_ok=False)
    video = output/'overlay24.mp4'
    sheet = Image.new('RGB',(960,3*(270+24)), '#151515')
    sheet_count = rendered = 0
    gallery_ids=sorted(track_index)[:64]
    gallery=Image.new('RGB',(960,max(1,math.ceil(len(gallery_ids)/4))*300),'#151515')
    gallery_seen=set()
    label_font=_font(max(18,round(info['width']*18/1100)))
    header_font=_font(max(12,round(info['width']*11/1100)))
    header_height=max(46,round(info['width']*30/1100))
    previous_index = None
    synthetic = 'SYNTHETIC FIXTURE | ' if manifest.get('synthetic') else ''
    reader = decode_video(source, manifest['source']['id'])
    try:
        with av.open(str(video),'w') as container:
            stream = container.add_stream('libx264',rate=24)
            stream.width, stream.height = info['width'],info['height']
            stream.pix_fmt = 'yuv420p' if info['width']%2 == 0 and info['height']%2 == 0 else 'yuv444p'
            # Avoid panorama-sized lookahead/thread buffers on small CPU runtimes.
            stream.options = {'crf':'18','preset':'veryfast','tune':'zerolatency'}
            stream.codec_context.thread_count=2
            stream.codec_context.time_base = Fraction(1,24)
            focus_container=focus_stream=None
            if focus_track_id is not None:
                focus_container=av.open(str(output/'focus24.mp4'),'w')
                focus_stream=focus_container.add_stream('libx264',rate=24)
                focus_stream.width,focus_stream.height=960,540
                focus_stream.pix_fmt='yuv420p';focus_stream.options={'crf':'18','preset':'veryfast','tune':'zerolatency'}
                focus_stream.codec_context.thread_count=2
                focus_stream.codec_context.time_base=Fraction(1,24)
            for frame in reader:
                if frame.index >= limit:
                    break
                if frame.index in coverage and coverage[frame.index] != 'unsampled_by_sampler':
                    previous_index = frame.index
                exact = previous_index == frame.index
                use = previous_index if previous_index is not None and (frame.index-previous_index)/24 <= hold_seconds else None
                image = frame.image.to_image().convert('RGB')
                draw = ImageDraw.Draw(image)
                saved = tracks.get(use,[])
                needs_raw=focus_track_id is not None or (exact and any(r['local_track_id'] in gallery_ids and r['local_track_id'] not in gallery_seen for r in saved))
                raw=image.copy() if needs_raw else None
                keys = {tuple(round(v,4) for v in row['bbox_xyxy']) for row in saved}
                for row in proposals.get(use,[]) if show_proposals else ():
                    if tuple(round(v,4) for v in row['bbox_xyxy']) in keys:
                        continue
                    box = row['bbox_xyxy']
                    draw.rectangle(box,outline='#ffb000',width=2)
                    draw.text((box[0],max(0,box[1]-12)),f"proposal {row['score']:.2f}",fill='#ffb000')
                occupied=[]
                for row in sorted(saved,key=lambda r:r['local_track_id']):
                    box = row['bbox_xyxy']
                    tid=row['local_track_id'];color=_track_color(tid)
                    draw.rectangle(box,outline=color,width=max(3,info['width']//600))
                    _badge(draw,box,f'T{tid}',color,label_font,occupied,image.width,image.height,header_height)
                    if exact and tid in gallery_ids and tid not in gallery_seen:
                        slot=gallery_ids.index(tid);x=(slot%4)*240;y=(slot//4)*300
                        margin=max(10,(box[3]-box[1])*.15)
                        crop=raw.crop((max(0,int(box[0]-margin)),max(0,int(box[1]-margin)),min(raw.width,math.ceil(box[2]+margin)),min(raw.height,math.ceil(box[3]+margin))))
                        crop.thumbnail((220,230));gallery.paste(crop,(x+(240-crop.width)//2,y+6))
                        gd=ImageDraw.Draw(gallery)
                        gd.text((x+8,y+242),f'T{tid}',font=_font(24),fill=color)
                        gd.text((x+8,y+274),f'First seen {frame.timestamp_s:.2f}s',font=_font(14),fill='white')
                        gallery_seen.add(tid)
                state = ('SAMPLED' if exact else f'HELD from frame {use} - no new inference' if use is not None else 'UNSAMPLED - no overlay')
                draw.rectangle((0,0,image.width,min(header_height,image.height)),fill='black')
                draw.text((6,3),f'{synthetic}{frame.timestamp_s:.3f}s | {state}',font=header_font,fill='white')
                draw.text((6,header_height//2),'T = temporary track ID, NOT jersey number or confirmed name',font=header_font,fill='white')
                if focus_container is not None:
                    focus=Image.new('RGB',(960,540),'black')
                    target=next((r for r in saved if r['local_track_id']==focus_track_id),None)
                    fd=ImageDraw.Draw(focus)
                    if target is not None:
                        x1,y1,x2,y2=target['bbox_xyxy'];cx=(x1+x2)/2;cy=(y1+y2)/2
                        cw=max(160,(x2-x1)*5,(y2-y1)*2.5*16/9);ch=cw*9/16
                        region=(math.floor(cx-cw/2),math.floor(cy-ch/2),math.ceil(cx+cw/2),math.ceil(cy+ch/2))
                        focus=raw.crop(region).resize((960,540));fd=ImageDraw.Draw(focus)
                        sx=960/(region[2]-region[0]);sy=540/(region[3]-region[1])
                        fd.rectangle(((x1-region[0])*sx,(y1-region[1])*sy,(x2-region[0])*sx,(y2-region[1])*sy),outline=_track_color(focus_track_id),width=4)
                    fd.rectangle((0,0,960,84),fill='black')
                    fd.text((10,6),f'T{focus_track_id} | {frame.timestamp_s:.3f}s | '+(state if target else 'NOT OBSERVED - no identity inferred'),font=_font(22),fill='white')
                    fd.text((10,44),'Zoomed recorded track, not a jersey number or confirmed player',font=_font(20),fill='white')
                    ff=av.VideoFrame.from_image(focus);ff.pts=rendered;ff.time_base=Fraction(1,24)
                    for packet in focus_stream.encode(ff):focus_container.mux(packet)
                if frame.index in sheet_indices:
                    thumb=image.copy();thumb.thumbnail((480,270))
                    x=(sheet_count%2)*480;y=(sheet_count//2)*294
                    sheet.paste(thumb,(x,y))
                    ImageDraw.Draw(sheet).text((x+4,y+272),f"Frame {frame.index} | {frame.timestamp_s:.3f}s",fill='white')
                    sheet_count += 1
                encoded = av.VideoFrame.from_image(image)
                encoded.pts, encoded.time_base = rendered,Fraction(1,24)
                for packet in stream.encode(encoded):container.mux(packet)
                rendered += 1
            for packet in stream.encode():container.mux(packet)
            if focus_container is not None:
                for packet in focus_stream.encode():focus_container.mux(packet)
    finally:
        reader.close()
        if 'focus_container' in locals() and focus_container is not None:
            focus_container.close()
    if rendered != limit:
        raise ValueError('Source ended before recorded coverage; preview incomplete')
    sheet.save(output/'contact-sheet.jpg')
    gallery.save(output/'players.jpg')
    (output/'track-index.json').write_text(json.dumps({'run_id':manifest['run_id'],
        'tracks_are_roster_identities':False,'gallery_ids':gallery_ids,
        'gallery_limit':64,'tracks':[track_index[tid] for tid in sorted(track_index)]},indent=2)+'\n',encoding='utf-8')
    result={'schema':'touchline.visual-preview.v1','run_id':manifest['run_id'],
        'source_sha256':original_hash,'checkpoint':manifest['detector'].get('checkpoint'),
        'detector':manifest['detector'],'run_status':manifest['status'],
        'frames':rendered,'fps':24,'duration_s':rendered/24,
        'hold_seconds':hold_seconds,'held_overlays_are_new_evidence':False,
        'render_scope':'full_analyzed_run' if full_run else 'bounded_preview',
        'tracks_are_roster_identities':False,'audio':'video-only preview',
        'focus_track_id':focus_track_id,'show_proposals':show_proposals,
        'players_gallery_sha256':sha256_file(output/'players.jpg'),
        'track_index_sha256':sha256_file(output/'track-index.json'),
        'overlay_sha256':sha256_file(video),'contact_sheet_sha256':sha256_file(output/'contact-sheet.jpg')}
    if focus_track_id is not None:result['focus_sha256']=sha256_file(output/'focus24.mp4')
    (output/'preview.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
