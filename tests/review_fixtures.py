"""Explicit synthetic local-review inputs; never field or server-auth evidence."""
def review_entry(run_id, track_id, player_id='P07', revision=1):
    return {'run_id':run_id,'local_track_id':track_id,'player_id':player_id,
            'confirmed':True,'reviewed_by':'fixture-coach','revision':revision,
            'player_name':'Alex','is_target':True}


def review_ledger(run_id, track_id, player_id='P07', revision=1):
    return {'run_id':run_id,'associations':{
        str(track_id):review_entry(run_id,track_id,player_id,revision)}}
