"""Compare two model proposal sets; disagreements are never ground-truth errors."""
from .contracts import Box

FRAME_INDICES=tuple(range(0,480,60))


def compare_boxes(student,teacher,width,height):
    if len(student)>1000 or len(teacher)>256:raise ValueError('Proposal count exceeds review bound')
    a=[Box(*b['bbox_xyxy']) for b in student];b=[Box(*v['bbox_xyxy']) for v in teacher]
    if any(not box.fits(width,height) for box in a+b):raise ValueError('Proposal geometry changed')
    pairs=sorted(((x.iou(y),i,j) for i,x in enumerate(a) for j,y in enumerate(b)),reverse=True)
    used_a=set();used_b=set();matched=[]
    for iou,i,j in pairs:
        if iou>=.5 and i not in used_a and j not in used_b:
            used_a.add(i);used_b.add(j);matched.append({'student':i,'sam':j,'iou':iou})
    return {'matched':matched,'sam_only':[j for j in range(len(b)) if j not in used_b],
        'student_only':[i for i in range(len(a)) if i not in used_a],
        'ground_truth_available':False,'identity_accuracy':None,'training_eligible':False,
        'review_status':'pending','promotion':'not performed'}
