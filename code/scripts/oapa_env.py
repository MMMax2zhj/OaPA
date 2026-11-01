"""Seeded environment witness for OaPA; no dataset or model IO."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import sklearn
import scipy
import joblib
import threadpoolctl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.strict and args.device.startswith('cuda') and torch.cuda.get_device_name(0)!='NVIDIA A800 80GB PCIe':
        raise RuntimeError('Strict witness requires NVIDIA A800 80GB PCIe')
    values=[]
    for _ in range(2):
        torch.manual_seed(20261010)
        layer=torch.nn.Linear(30,2).to(args.device)
        x=torch.randn(16,32,30,device=args.device,requires_grad=True)
        loss=layer(x).square().mean();loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        values.append((loss.detach().cpu(),x.grad.detach().cpu()))
    torch.testing.assert_close(values[0],values[1],atol=0,rtol=0)
    result={'passed':True,'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
            'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,
            'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__,
            'device':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
            'threads':4,'TF32':False,'cudnn_deterministic':True,
            'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'exact_seeded_repeat':True,'seed':20261010,'kernel_loss':float(values[0][0])}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
