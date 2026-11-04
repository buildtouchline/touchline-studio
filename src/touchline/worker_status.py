"""Bounded worker snapshots for reconnectable controllers; no provider dependencies."""
import json,math

STAGES={'input','checks','precision_parity','analysis','render','complete'}
STATUSES={'started','progress','heartbeat','passed','saved','complete','failed'}
FIELDS={'stage','status','frames','selected_frames','tests','ids','error_category','run_id','gpu_memory_mib','event_seq','worker_epoch'}

def parse_worker_status(text):
    if not isinstance(text,str) or not 0<len(text.encode())<=16384:raise ValueError('Bounded worker status required')
    item=json.loads(text)
    if not isinstance(item,dict) or set(item)-FIELDS or item.get('stage') not in STAGES or item.get('status') not in STATUSES:raise ValueError('Unexpected worker status schema')
    for key in ('frames','selected_frames','tests','ids','gpu_memory_mib','event_seq'):
        if key in item and (type(item[key]) is not int or not 0<=item[key]<=100000000):raise ValueError('Invalid worker counter')
    if not 1<=item.get('event_seq',0) or type(item.get('worker_epoch')) not in (int,float) or not math.isfinite(item['worker_epoch']) or item['worker_epoch']<=0:raise ValueError('Versioned worker status required')
    if item.get('ids',0)>24:raise ValueError('Session ID capacity exceeded')
    for key in ('error_category','run_id'):
        if key in item and (not isinstance(item[key],str) or not 1<=len(item[key])<=100 or any(not(c.isalnum() or c in '_-') for c in item[key])):raise ValueError('Invalid worker identifier')
    return item
