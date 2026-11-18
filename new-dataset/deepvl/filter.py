"""ReAqROVIO no-feature inertial core port, with two heading observations.

BSD-3-Clause upstream notices and original source are retained in vendor/.
Nominal: world p[3], NEGATIVE body velocity u[3], ba[3], bg[3], qWM xyzw[4].
Error: dp, du, dba, dbg, LEFT world-frame dtheta (15).
Ported from ImuPrediction.hpp, VelocityUpdate.hpp, BaroUpdate.hpp. Removing
camera/feature states does not replace this core by our attitude-aided ESKF.
"""
import numpy as np
from scipy.spatial.transform import Rotation

GRAVITY=np.array([0.,0.,-9.806])
# cfg/rcm_equirefrac/rovio_rcm_online.info at pinned ReAqROVIO commit.
NOISE=np.repeat([1e-4,1e-7,1.84e-6,2.06e-10,1.09e-7],3)
INITIAL=np.repeat([1e-4,.1,5e-4,2e-8,.1],3)
VELOCITY_GATE=7.689997599999999
DEPTH_GATE=12.6511204
DEPTH_VARIANCE=.5


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def left_jacobian(v):
    a=np.linalg.norm(v);k=skew(v)
    if a<1e-6:return np.eye(3)+.5*k+(1/6)*k@k
    return np.eye(3)+(1-np.cos(a))/a**2*k+(a-np.sin(a))/a**3*k@k


def inject(state,delta):
    out=state.copy();out[:12]+=delta[:12]
    out[12:]= (Rotation.from_rotvec(delta[12:])*Rotation.from_quat(state[12:])).as_quat()
    return out


def difference(a,b):
    return np.r_[a[:12]-b[:12],(Rotation.from_quat(a[12:])*Rotation.from_quat(b[12:]).inv()).as_rotvec()]


def propagation(state,acc,gyro,dt=.005):
    r=Rotation.from_quat(state[12:]);R=r.as_matrix();u=state[3:6];domega=dt*(gyro-state[9:12])
    out=state.copy();out[:3]-=dt*R@u
    out[3:6]=(np.eye(3)-skew(domega))@u-dt*(acc-state[6:9]+R.T@GRAVITY)
    out[12:]=(r*Rotation.from_rotvec(domega)).as_quat()
    F=np.eye(15);F[:3,3:6]=-dt*R;F[:3,12:]=dt*skew(R@u)
    F[3:6,3:6]=np.eye(3)-skew(domega);F[3:6,6:9]=dt*np.eye(3)
    F[3:6,9:12]=-dt*skew(u);F[3:6,12:]=-dt*R.T@skew(GRAVITY)
    F[12:,9:12]=-dt*R@left_jacobian(domega)
    G=np.eye(15)*np.sqrt(dt);G[3:6,12:]=skew(u)*np.sqrt(dt)
    G[12:,12:]=R@left_jacobian(domega)*np.sqrt(dt)
    return out,F,G


def heading_observation(state):
    R=Rotation.from_quat(state[12:]).as_matrix();yaw=np.arctan2(R[1,0],R[0,0])
    Z=Rotation.from_euler('z',yaw).as_matrix();vw=-R@state[3:6]
    h=(Z.T@vw)[:2]
    first=R[:,0];den=first[0]**2+first[1]**2
    if den<1e-8:raise FloatingPointError('Heading undefined near vertical pitch')
    dyaw=np.array([-first[1],first[0],0.])@(-skew(first))/den
    H=np.zeros((2,15));H[:,3:6]=(-Z.T@R)[:2]
    H[:,12:]=(Z.T@(-skew(vw)-np.outer(skew([0.,0.,1.])@vw,dyaw)))[:2]
    return h,H


def measurement_update(state,P,value,pred,H,variance,gate):
    residual=value-pred;R=np.atleast_2d(variance);S=H@P@H.T+R
    nis=float(residual@np.linalg.solve(S,residual))
    if not np.isfinite(nis):raise FloatingPointError('Nonfinite innovation')
    if nis>gate:return state,P,False,nis
    K=np.linalg.solve(S,H@P).T;delta=K@residual
    out=inject(state,delta);A=np.eye(15)-K@H
    covariance=A@P@A.T+K@R@K.T
    reset=np.eye(15);reset[12:,12:]=left_jacobian(delta[12:])
    covariance=reset@covariance@reset.T
    return out,(covariance+covariance.T)*.5,True,nis


def anchor(state,P,yaw,xy):
    out=state.copy();rot=Rotation.from_quat(state[12:]);old=rot.as_euler('xyz')[2]
    z=Rotation.from_euler('z',yaw-old);Z=z.as_matrix()
    out[:3]=Z@state[:3];out[:2]=xy;out[12:]=(z*rot).as_quat()
    J=np.eye(15);J[:3,:3]=Z;J[12:,12:]=Z
    return out,J@P@J.T


def replay(record,ends,means,variances,anchor_yaw,anchor_xy):
    n=len(record.time);state=np.zeros(16)
    a=np.median(record.accel[:min(200,n)],axis=0)
    tilt=[np.arctan2(a[1],a[2]),np.arctan2(-a[0],np.hypot(a[1],a[2])),0.]
    state[12:]=Rotation.from_euler('xyz',tilt).as_quat();P=np.diag(INITIAL)
    states=np.empty((n,16));diagonal=np.empty((n,15));events=[];j=0
    for i in range(n):
        if i:
            state,F,G=propagation(state,record.accel[i],record.gyro[i],record.time[i]-record.time[i-1])
            P=F@P@F.T+(G*NOISE)@G.T;P=(P+P.T)*.5
        if i==1600:state,P=anchor(state,P,float(anchor_yaw),anchor_xy)
        if j<len(ends) and i==ends[j]:
            h,H=heading_observation(state)
            state,P,ok,nis=measurement_update(state,P,means[j],h,H,np.diag(variances[j]),VELOCITY_GATE)
            events.append(dict(time=float(record.time[i]),kind='network_xy',accepted=ok,nis=nis));j+=1
        # Same 50Hz depth timestamps as the existing processed-CSV runtime.
        if i%4==3:
            H=np.zeros((1,15));H[0,2]=1
            state,P,ok,nis=measurement_update(state,P,np.array([-(record.depth[i]-record.depth[0])]),
                      state[2:3],H,np.array([[DEPTH_VARIANCE]]),DEPTH_GATE)
            events.append(dict(time=float(record.time[i]),kind='depth',accepted=ok,nis=nis))
        if not np.isfinite(state).all() or not np.isfinite(P).all():raise FloatingPointError(f'Filter diverged at {i}')
        if i%200==0 and np.linalg.eigvalsh(P).min() < -1e-10:raise FloatingPointError('Covariance lost PSD')
        states[i]=state;diagonal[i]=np.diag(P)
    rotation=Rotation.from_quat(states[:,12:]);world_velocity=-rotation.apply(states[:,3:6])
    projection=np.column_stack((states[:,:3],world_velocity,rotation.as_euler('xyz')[:,2]))
    return projection,states,diagonal,events
