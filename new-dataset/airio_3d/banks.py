"""Whole-split device banks, explicit no-cross-segment window indices."""
import numpy as np
import torch
import contracts as c
import pypose as pp


class Bank:
    def __init__(self,split,stage,device):
        self.device=device;keys=('acc','gyro','position','velocity','quaternion','body_velocity','gravity_body')
        parts={k:[] for k in keys};starts=[];self.records=[];offset=0
        stride=c.CONFIG[stage]['train_stride' if split=='train' else 'val_stride']
        for path in c.split_paths()[split]:
            r=c.load_record(path);n=len(r['time'])
            if n<=1000:continue
            candidate=np.arange(0,n-1000,stride,dtype=int);end=r['time'][candidate+999]
            valid=np.ones(len(candidate),bool)
            # Retain the pre-existing common quality exclusion for offline
            # windows. No RPM is an online input or online gating signal.
            for lo,hi in r['excluded_end_intervals']:valid&=~((end>=lo-1e-9)&(end<=hi+1e-9))
            candidate=candidate[valid];starts.extend((candidate+offset).tolist())
            self.records.append(dict(source=str(path),windows=len(candidate),offset=offset,samples=n))
            for k in keys:parts[k].append(r[k])
            offset+=n
        if not starts:raise ValueError('No valid windows')
        self.arrays={k:torch.as_tensor(np.concatenate(v).astype('float32'),device=device) for k,v in parts.items()}
        self.starts=torch.tensor(starts,device=device,dtype=torch.long)
        self.span=torch.arange(1000,device=device)

    def __len__(self):return len(self.starts)

    def batch(self,index,stage):
        ix=self.starts[index,None]+self.span[None,:];initial=self.starts[index]
        a=self.arrays;acc=a['acc'][ix];gyro=a['gyro'][ix]
        if stage=='airimu':
            pad=a['gravity_body'][initial,None].expand(-1,9,-1)
            data=dict(acc=torch.cat((pad,acc),1),gyro=torch.cat((torch.zeros_like(pad),gyro),1),
                      dt=torch.full_like(acc[:,:,:1],c.DT),rot=pp.SO3(a['quaternion'][ix]))
            init=dict(pos=a['position'][initial,None],vel=a['velocity'][initial,None],rot=pp.SO3(a['quaternion'][initial,None]))
            label=dict(gt_pos=a['position'][ix+1],gt_vel=a['velocity'][ix+1],gt_rot=pp.SO3(a['quaternion'][ix+1]))
            return data,init,label
        label_ix=self.starts[index,None]+torch.arange(1001,device=self.device)[None]
        return dict(acc=acc,gyro=gyro),pp.SO3(a['quaternion'][ix]).Log().tensor(),a['body_velocity'][label_ix]
