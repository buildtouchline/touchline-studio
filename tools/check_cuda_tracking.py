"""OPERATOR ONLY: free-Colab synthetic CUDA parity check, not a quality benchmark.

Run explicitly after installation: python tools/check_cuda_tracking.py
Does not load weights, read footage/credentials, or contact a provider.
"""
import json


def main():
    import numpy as np
    from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
    from touchline.cuda_motion import CudaMotionPredictor
    from touchline.byte_tracking import ByteTrackTracker
    from touchline.contracts import Box,Detection
    reference=KalmanFilter();gpu=CudaMotionPredictor(reference)
    state=[reference.initiate(np.asarray([100+i*20,200,.4,80+i],dtype=float)) for i in range(32)]
    mean=np.stack([m for m,_ in state]);cov=np.stack([c for _,c in state])
    expected_mean,expected_cov=reference.multi_predict(mean.copy(),cov.copy())
    actual_mean,actual_cov=gpu.multi_predict(mean,cov)
    np.testing.assert_allclose(actual_mean,expected_mean,rtol=1e-4,atol=1e-3)
    np.testing.assert_allclose(actual_cov,expected_cov,rtol=1e-4,atol=1e-3)
    cpu_tracker=ByteTrackTracker(12);gpu_tracker=ByteTrackTracker(12,prediction_device='cuda')
    for i in range(24):
        detections=[] if i in (10,11) else [Detection(Box(10+i*2,50,40+i*2,120),.9)]
        a=cpu_tracker.update(i/12,detections);b=gpu_tracker.update(i/12,detections)
        assert [r['local_track_id'] for r in a]==[r['local_track_id'] for r in b], 'CPU/CUDA association differs on fixture'
    print(json.dumps({'result':'synthetic_cuda_parity_passed','tracker_runtime':gpu_tracker.runtime_stats(),
        'real_video_accuracy_verified':False,'performance_improvement_verified':False},indent=2))


if __name__=='__main__':main()
