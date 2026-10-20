"""Isolated timed replay of StarPilot ModelState, VisionIPC and publication."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

def main():
    if Path('/AGNOS').exists():
        raise RuntimeError('This isolated v9 replay benchmark is host-only, not for comma hardware')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--warmup', type=float, default=30)
    p.add_argument('--scratch', default='/tmp/starpilot-onexgpu-bench/timed')
    a = p.parse_args()
    root = Path(a.scratch)
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    prefix = 'onexbench_' + str(os.getpid())
    Path('/dev/shm/msgq_' + prefix).mkdir(exist_ok=True)
    os.environ.update(OPENPILOT_PREFIX=prefix, PARAMS_ROOT=str(root/'params'),
        COMMA_CACHE=str(root/'downloads'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        XDG_CACHE_HOME='/tmp/starpilot-onexgpu-bench')
    import av
    import numpy as np
    from openpilot.system.hardware.hw import Paths
    Paths.comma_home = staticmethod(lambda: str(root/'state'))
    from cereal import messaging, log
    from msgq.visionipc import VisionIpcClient, VisionIpcServer, VisionStreamType
    from openpilot.selfdrive.modeld import modeld
    from openpilot.tools.lib.logreader import LogReader
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    from openpilot.common.transformations.camera import DEVICE_CAMERAS
    from openpilot.common.transformations.model import get_warp_matrix
    from tinygrad import helpers
    from tinygrad.device import Device
    helpers.DEV.value = 'CPU:LLVM;USB+AMD:LLVM'
    os.environ['DEV'] = 'CPU:LLVM;USB+AMD:LLVM'
    modeld.get_tg_input_devices = lambda *_a, **_k: {'WARP_DEV':'CPU','QUEUE_DEV':'AMD'}
    data = Path('/tmp/starpilot-onexgpu-bench/replay')
    meta = {}
    for msg in LogReader(str(data/'rlog.zst')):
        if msg.which() in ('deviceState','roadCameraState','liveCalibration','carState'):
            meta.setdefault(msg.which(), msg)
        if len(meta) == 4: break
    dc = DEVICE_CAMERAS[(str(meta['deviceState'].deviceState.deviceType),
                         str(meta['roadCameraState'].roadCameraState.sensor))]
    cal = np.array(meta['liveCalibration'].liveCalibration.rpyCalib)
    tfms = [get_warp_matrix(cal,dc.fcam.intrinsics,False).astype(np.float32),
            get_warp_matrix(cal,dc.ecam.intrinsics,True).astype(np.float32)]
    speed = float(meta['carState'].carState.vEgo)
    w,h = 1928,1208
    stride,yh,_,size = get_nv12_info(w,h)
    modeld._set_hcq_wait_timeout(modeld.BIG_MODEL_LOAD_WAIT_TIMEOUT_MS)
    model = modeld.ModelState(w,h,True,model_id_override='local-onexgpu-bench',
        model_path_override=Path('/tmp/starpilot-onexgpu-bench/combined-amd.pkl'),
        model_version_override='v9',write_model_version=False,force_external_gpu=True)
    assert model.uses_external_gpu and type(Device['AMD'].iface).__name__ == 'USBIface'
    model.warmup()
    modeld._set_hcq_wait_timeout(modeld.BIG_MODEL_RUN_WAIT_TIMEOUT_MS)
    print('MODEL_READY',Device['AMD'].arch,flush=True)
    kinds = [VisionStreamType.VISION_STREAM_ROAD,VisionStreamType.VISION_STREAM_WIDE_ROAD]
    server = VisionIpcServer('onex_bench')
    for kind in kinds:
        server.create_buffers_with_sizes(kind,8,w,h,size,stride,stride*yh)
    server.start_listener()
    clients = [VisionIpcClient('onex_bench',kind,True) for kind in kinds]
    for client in clients:
        deadline = time.monotonic()+10
        while not client.connect(False):
            if time.monotonic()>deadline: raise TimeoutError('VisionIPC connection')
            time.sleep(0.05)
    pm = messaging.PubMaster(['modelV2','drivingModelData','cameraOdometry','chestnutState'])
    sub = messaging.sub_sock('modelV2',timeout=1000,conflate=False)
    telemetry = modeld.ChestnutState(pm,True)
    stages = {}
    def timed(name,fn):
        def call(*args,**kwargs):
            start=time.monotonic_ns()
            try: return fn(*args,**kwargs)
            finally: stages[name]=(time.monotonic_ns()-start)/1e6
        return call
    model.warp_enqueue=timed('warp_ms',model.warp_enqueue)
    model.run_policy=timed('policy_enqueue_ms',model.run_policy)
    send_telemetry=timed('telemetry_ms',telemetry.send)
    toggles = modeld.get_starpilot_toggles(messaging.SubMaster(['starpilotPlan']))
    pubstate = modeld.PublishState()
    action = log.ModelDataV2.Action()
    total,warm = round((a.seconds+a.warmup)*20),round(a.warmup*20)
    sent_at,state = {},dict(sent=0,late=0,error=None)
    done,stop = threading.Event(),threading.Event()
    def frames(name):
        while not stop.is_set():
            with av.open(str(data/name)) as video:
                video.streams.video[0].thread_type = 'AUTO'
                for frame in video.decode(video=0):
                    yield frame.to_ndarray(format='nv12')
    def source():
        try:
            videos = [frames(n) for n in ('fcamera.hevc','ecamera.hevc')]
            packed = [np.zeros(size,dtype=np.uint8) for _ in kinds]
            epoch = None
            for fid in range(1,total+1):
                if stop.is_set(): break
                for video,buf in zip(videos,packed):
                    frame = next(video)
                    buf[:stride*yh].reshape(yh,stride)[:h,:w] = frame[:h,:w]
                    buf[stride*yh:stride*(yh+h//2)].reshape(h//2,stride)[:,:w] = frame[h:,:w]
                if epoch is None: epoch = time.monotonic()+0.1
                target = epoch+(fid-1)/20
                stop.wait(max(0,target-time.monotonic()))
                if stop.is_set(): break
                if fid>warm and time.monotonic()-target>0.05: state['late']+=1
                stamp = time.monotonic_ns()
                sent_at[fid] = stamp
                for kind,buf in zip(kinds,packed): server.send(kind,buf,fid,stamp,stamp)
                state['sent'] = fid
        except BaseException: state['error'] = traceback.format_exc()
        finally: done.set()
    time.sleep(0.2)
    thread = threading.Thread(target=source,daemon=True)
    thread.start()
    rows,n,mismatch = [],0,0
    last,progress = time.monotonic(),time.monotonic()
    try:
        while not (done.is_set() and time.monotonic()-last>2):
            road = clients[0].recv()
            if road is None:
                if state['error']: raise RuntimeError(state['error'])
                continue
            fid = clients[0].frame_id
            wide = clients[1].recv()
            while wide is not None and clients[1].frame_id<fid: wide=clients[1].recv()
            if wide is None or clients[1].frame_id!=fid:
                mismatch+=1
                continue
            bufs,trans,inputs = modeld._runner_frame_args(model,road,wide,*tfms,
                np.zeros(8,dtype=np.float32),np.array([1,0],dtype=np.float32),
                0.25,0.45,action,speed,np.array([speed,0.25],dtype=np.float32))
            begin = time.monotonic_ns()
            stages.clear()
            out = model.run(bufs,trans,inputs,False,send_telemetry if n%2==0 else None)
            elapsed = (time.monotonic_ns()-begin)/1e9
            assert out is not None and model.uses_external_gpu
            for value in out.values():
                if isinstance(value,np.ndarray): assert np.isfinite(value).all()
            action = modeld.get_action_from_model(out,action,0.25,0.45,speed,
                model.mlsim,model.is_v9,model.is_v14,model.is_v15,toggles,0.1,0.3,
                is_v16=model.is_v16)
            msg,driving,pose = [messaging.new_message(s) for s in
                               ('modelV2','drivingModelData','cameraOdometry')]
            stamp = sent_at[fid]
            modeld.fill_model_msg(driving,msg,out,action,pubstate,
                fid,fid,fid,0.0,stamp,elapsed,True)
            modeld.fill_pose_msg(pose,out,fid,0,stamp,True)
            pm.send('modelV2',msg)
            pm.send('drivingModelData',driving)
            pm.send('cameraOdometry',pose)
            received = messaging.recv_one(sub)
            assert received is not None and received.modelV2.frameId==fid
            latency = (time.monotonic_ns()-stamp)/1e6
            last = time.monotonic()
            n+=1
            if fid>warm:
                rows.append(dict(frame_id=fid,execution_ms=elapsed*1000,latency_ms=latency,
                    warp_ms=stages.get('warp_ms',0),
                    policy_enqueue_ms=stages.get('policy_enqueue_ms',0),
                    telemetry_ms=stages.get('telemetry_ms',0)))
            if last-progress>10:
                print(json.dumps(dict(event='progress',sent=state['sent'],processed=n,
                    measured=len(rows),latency_ms=round(latency,2))),flush=True)
                progress=last
    finally:
        stop.set()
        thread.join(5)
    assert rows,'No measured outputs'
    lt=np.array([r['latency_ms'] for r in rows])
    ex=np.array([r['execution_ms'] for r in rows])
    expected=total-warm
    report=dict(scope='ModelState CPU warp + AMD vision/policy + VisionIPC + publication; not full manager/control stack',
        seconds=a.seconds,warmup_seconds=a.warmup,expected=expected,received=len(rows),
        missing=expected-len(rows),source_late_frames=state['late'],mismatched_pairs=mismatch,
        execution_p50_ms=float(np.percentile(ex,50)),execution_p99_ms=float(np.percentile(ex,99)),
        latency_p50_ms=float(np.percentile(lt,50)),latency_p99_ms=float(np.percentile(lt,99)),
        latency_max_ms=float(lt.max()),deadline_misses=int((lt>50).sum()),
        gpu=Device['AMD'].arch,interface=type(Device['AMD'].iface).__name__,error=state['error'])
    report['stage_p50_ms']={key:float(np.median([r[key] for r in rows]))
        for key in ('warp_ms','policy_enqueue_ms','telemetry_ms')}
    report['gpu_metrics']=telemetry.metrics
    report['passed']=(len(rows)==expected and not state['late'] and
                      not report['deadline_misses'] and not state['error'])
    (root/'frames.json').write_text(json.dumps(rows))
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['passed'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
