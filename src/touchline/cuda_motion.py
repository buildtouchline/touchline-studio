"""Explicit hybrid ByteTrack: batched CUDA motion prediction, CPU association/update."""
from time import perf_counter


class CudaMotionPredictor:
    """Instance-local replacement for the pinned backend's shared Kalman predictor.

    Implements x'=Fx and P'=FPF^T+Q in a single tensor batch. Correction,
    association and lifecycle remain the audited backend's CPU implementation.
    CUDA is mandatory: no silent device fallback or automatic weight download.
    """
    def __init__(self,reference):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('A compatible Colab CUDA Torch runtime is required') from exc
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA motion prediction requested, but CUDA is unavailable')
        self.torch=torch;self.reference=reference
        self.motion=torch.as_tensor(reference._motion_mat,dtype=torch.float32,device='cuda')
        if self.motion.device.type!='cuda':raise RuntimeError('Motion matrix is not on CUDA')
        self.calls=0;self.max_batch_size=0;self.wall_seconds=0.0

    def multi_predict(self,mean,covariance):
        if len(mean)==0:return mean,covariance
        torch=self.torch;started=perf_counter()
        with torch.inference_mode():
            m=torch.as_tensor(mean,dtype=torch.float32,device=self.motion.device)
            p=torch.as_tensor(covariance,dtype=torch.float32,device=self.motion.device)
            h=m[:,3];ones=torch.ones_like(h)
            pos=h*self.reference._std_weight_position
            vel=h*self.reference._std_weight_velocity
            noise=torch.stack((pos,pos,.01*ones,pos,vel,vel,.00001*ones,vel),dim=1)
            predicted_mean=m@self.motion.T
            predicted_cov=self.motion@p@self.motion.T+torch.diag_embed(noise.square())
            if not bool(torch.isfinite(predicted_mean).all()) or not bool(torch.isfinite(predicted_cov).all()):
                raise ValueError('Nonfinite CUDA motion state; stop rather than invent an identity')
            result=(predicted_mean.cpu().numpy(),predicted_cov.cpu().numpy())
        self.calls+=1;self.max_batch_size=max(self.max_batch_size,len(mean))
        self.wall_seconds+=perf_counter()-started
        return result

    def runtime_stats(self):
        return {'motion_prediction_device':str(self.motion.device),'prediction_tensor_dtype':'float32',
            'cuda_prediction_batches':self.calls,'max_prediction_batch_size':self.max_batch_size,
            'prediction_wall_seconds_including_transfers':self.wall_seconds,
            'gpu_name':self.torch.cuda.get_device_name(self.motion.device),
            'association_device':'cpu','kalman_correction_device':'cpu','full_gpu_tracker':False}
