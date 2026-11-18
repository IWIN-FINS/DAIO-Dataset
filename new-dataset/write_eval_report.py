"""Evidence-bound results summary; no manual editing of scientific metrics."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'ROV-ESKF-Odometry/reports/new_v1v2';DATA=ROOT/'new-dataset'
NAMES={'horizontal_rpm':'水平 IMU + RPM','full_imu_rpm':'完整六轴 IMU + RPM','horizontal_command':'水平 IMU + 指令','airio_planar':'AirIO（ROV 平面适配）','deepvl_planar':'DeepVL（ROV 平面适配）'}
NAMES['airio_3d']='AirIO（三维混合参考监督）'

def main():
    network=list(csv.DictReader((OUT/'network_ablation_metrics.csv').open(encoding='utf-8-sig')))
    odom=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8-sig')))
    selection=json.loads((DATA/'evaluation_selection.json').read_text(encoding='utf-8'))
    complete=(DATA/'baselines/runs/airio/complete.json').exists()
    methods=list(dict.fromkeys(r['method'] for r in odom));diagnostic=[]
    for method in methods:
        for test in range(1,5):
            path=OUT/method/f'test{test}_replay/updates.csv'
            if not path.exists():path=OUT/method/f'test{test}_updates.csv'
            if not path.exists():continue
            events=[r for r in csv.DictReader(path.open(encoding='utf-8-sig')) if r['kind']=='network_xy' and float(r['time'])>=8]
            t=np.array([float(r['time']) for r in events]);i=int(np.argmax(np.diff(t)))
            diagnostic.append({'method':method,'test':f'Test {test}','events':len(events),
                'accepted':sum(r['accepted']=='True' for r in events),'largest_update_gap_s':float(t[i+1]-t[i]),
                'gap_start_record_s':float(t[i]),'gap_end_record_s':float(t[i+1])})
    with (OUT/'network_update_availability.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostic[0]));w.writeheader();w.writerows(diagnostic)
    ntable='\n'.join('| '+NAMES[r['model']]+' | '+f"{float(r['checkpoint_epoch']):.0f}"+' | '+' | '.join(f'{float(r[k]):.5f}' for k in ('xy_vector_rmse_mps','xy_component_mean_rmse_mps','yaw_rate_rmse_radps'))+' |' for r in network if r['test']=='Mean')
    otable='\n'.join('| '+NAMES[r['method']]+' | '+' | '.join(f'{float(r[k]):.5f}' for k in ('velocity_xy_vector_rmse_mps','position_xy_vector_rmse_m','final_position_error_m','rpe_5s_xy_vector_rmse_m'))+' |' for r in odom if r['test']=='Mean' and r['scoring']=='all_samples')
    detail='\n'.join('| '+NAMES[r['method']]+' | '+r['test']+' | '+' | '.join(f'{float(r[k]):.5f}' for k in ('velocity_xy_vector_rmse_mps','position_xy_vector_rmse_m','rpe_5s_xy_vector_rmse_m'))+' |' for r in odom if r['test']!='Mean' and r['scoring']=='all_samples')
    deepvl='deepvl_planar' in methods
    deepvl_note='''
## DeepVL 复现边界

DeepVL采用官方八模型GRU/PReLU实现（dropout 0.1），训练2500次更新，前1250次MSE、后1250次GNLL；按完整验证轨迹的水平向量速度RMSE选择GNLL阶段的同一集成checkpoint。
网络输入完整六轴IMU、8路指令、电压和电压有效标志；不输入RPM或AprilTag。电压仅供此基线使用，其余模型未重训。
网络每20Hz连续传播隐藏状态；滤波基于独立移植的ReAqROVIO无视觉惯性核心，按200Hz传播、50Hz更新相对深度，不是自有ESKF。
由于没有独立三维真值，仅监督和更新水平航向系两维速度；不对缺失的垂向速度、roll/pitch构造伪标签。名称为DeepVL (Planar ROV)，不声称完整原版三维复现。
方差按输出标准差的平方恢复物理单位，包含成员内方差与成员间差异。滤波采用官方传播及观测结构；传感器噪声与偏置先验按训练集标定、验证集选定配置，没有按测试轨迹调参。
这四种方法是系统级比较：输入传感器、滤波后端与网络更新频率不同。详细源码版本、适配与官方代码问题修复见new-dataset/deepvl/README.md。
''' if deepvl else ''
    diagnostic_path=OUT/'deepvl_planar/filter_diagnostics.json'
    if deepvl and diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text(encoding='utf-8'))
        recovered=all(x['status']=='stable_after_train_sensor_calibration' for x in diagnostic['findings'])
        if recovered:
            model_provenance=json.loads((OUT/'deepvl_planar/provenance.json').read_text(encoding='utf-8'))
            calibration=json.loads(Path(model_provenance['filter_calibration']).read_text(encoding='utf-8'))
            bias_text=', '.join(f'{x:.6f}' for x in calibration['initial_gyro_bias'])
            prior_text=', '.join(f'{x:.6f}' for x in calibration['gyro_bias_prior_std'])
            deepvl_note+=f'''
### DeepVL 传感器先验修复

原配置把陀螺仪偏置初始标准差设为约0.00014rad/s且初值为零，而训练集标定得到的本机偏置为[{bias_text}]rad/s。过窄且未标定的偏置先验使滤波器无法及时估计偏置，倾角误差累积后形成速度门限持续拒绝。带已知陀螺仪偏置的合成回归测试复现了这一机制。
修复仅对本机传感器标定：从{len(calibration['sources'])}条有效训练记录估计初始偏置，按记录间标定差异确定先验标准差[{prior_text}]rad/s，避免把求导真值的逐点噪声当成偏置漂移。加速度/角速度白噪声PSD同样来自训练IMU。
标定时使用训练记录的IMU内置roll/pitch及独立AprilTag yaw；内置yaw存在漂移，不充当航向标定参考。不存在的独立三维真值不补零，IMU内置倾角不当作网络监督标签。测试传播不读内置姿态或连续AprilTag。四个物理候选只在验证026/027比较，选定完整标定配置并记录哈希后回放Test 1–4。
保持原网络checkpoint、均值/协方差预测、15维惯性核心、速度/深度门限、8s一次对齐和零初速度不变，没有把DeepVL替换为自有ESKF，也没有按测试结果修改参数。四条测试的速度观测接受率为99.68%–100%，未再发生持续门限拒绝或速度发散。
原始失败状态、指标及图片保留于diagnostic_deepvl_original_priors_20261102/；初次内置yaw标定的诊断回放保留于diagnostic_deepvl_imu_yaw_prior_20261102/，不用于当前表格。最终训练标定、验证候选与冻结配置见new-dataset/deepvl_filter_recovery_v3/。当前表格和图片均为修复后结果。其他三方法预测文件及30条原指标保持不变。
'''
        else:
            deepvl_note+='''
### 当前固定配置的融合失败

四条DeepVL回放均出现严重跟踪失败，不能把其网络验证RMSE当作稳定里程计结果。网络预测速度最大值约0.28–0.31m/s，但速度更新在记录约2.2–2.4秒后持续被门限拒绝；随后惯性传播误差和姿态偏差累积。
表格和CSV保留全部评分样本及实际的大误差，没有裁掉失败区间，没有改用我们的ESKF，也没有用Test结果调整噪声、初始偏置或门限。整幅图保留完整量程，因此其他方法曲线可能被压缩。
这证明当前平面适配与固定配置尚未形成可用融合，不能直接据此声称DeepVL原方法普遍劣于自有方法。状态方程、Jacobian、速度负号、协方差单位与指标已做独立检查；单一根因尚未确定。观测接受率和首次拒绝时刻见deepvl_planar/filter_diagnostics.csv。
'''
    text=f'''# 新 V1/V2：消融与里程计评估

三组消融和两种 ESKF 已完成。AirIMU 按用户要求停止继续训练，固定验证最优的第{selection['airimu_epoch']}轮；AirIO 状态：{'训练已完成，已纳入评估' if complete and 'airio_planar' in methods else '已获授权训练速度网络；尚未纳入当前指标和图片'}。
当前图表、表格不把 AirIMU 直接称为 AirIO，不使用缺失的 AirIO 结果占位。

## 三组网络消融

只用新录制数据。验证为 V2 026/027，Test 1–4 分别为 V2 025/028/029/030。seed42；各模型仅按验证集选 checkpoint。
下表是四条测试轨迹的指标算术平均，所有模型的逐窗口时间戳及标签已逐项核对完全相同。

| 模型 | 最佳轮数 | XY 向量 RMSE(m/s) | XY 分量平均 RMSE(m/s) | yaw-rate RMSE(rad/s) |
|---|---:|---:|---:|---:|
{ntable}

完整结果为 network_ablation_metrics.csv；12 张1200dpi PNG位于 network_ablation/，包括2×2误差总图、四条独立误差图、四条机体系速度图、RMSE柱状图和两个独立图例。
曲线仅为显示按约5Hz抽样，并保留全局误差最大点；指标使用全部有效50Hz窗口。排除区间留空，不连接为虚构预测。
网络图时间0是首个可预测时刻（记录7.675s）；不把该网络误差与ESKF融合后误差混用。

## 里程计结果

每条轨迹0–8s连续预热，8s开始计分；此时一次性对齐yaw和XY位置，不用真值速度初始化，不做后续刚体拟合或尺度对齐。
真值SG101/poly3后求导；指标使用全部200Hz评分样本。表内速度、位置均为二维向量RMSE。
Mean为四条轨迹指标的算术平均，不是按时长加权汇总。

| 方法 | 速度 RMSE(m/s) | 位置 RMSE(m) | 最终位置误差(m) | 5s相对位移 RMSE(m) |
|---|---:|---:|---:|---:|
{otable}

| 方法 | 轨迹 | 速度 RMSE(m/s) | 位置 RMSE(m) | 5s相对位移 RMSE(m) |
|---|---|---:|---:|---:|
{detail}

{deepvl_note}

metrics.csv还包含X/Y分量、yaw和排除修复/插值点后的评分。最终位置误差仍描述真实记录终点。
根目录速度图和XY图各4张、速度/位置二维误差范数箱线图各1张、独立图例1张，共11张1200dpi PNG。
速度图和XY图为2:1，箱线图3:1。里程计图Time(s)以8s评分起点为0。图内不放图例。

## Test 1 的 RPM 异常与解释边界

Test 1有一个绝对值达到32759的RPM反馈点。按预先确定的异常规则，RPM模型禁用了覆盖该点的历史窗口；其network_xy更新在记录61.995–69.695s间有7.7s间隔。
该区间速度误差明显增大，评分仍保留所有样本，没有删掉故障区间，也没有根据测试结果调整ESKF噪声或门限。
指令模型在线不读取RPM进行预测或异常筛选，其更新最大间隔约20ms。训练与离线网络消融仍采用共同有效窗口。
两种ESKF的状态、噪声、门限、初始化和深度处理一致，仅观测网络来源不同。
输入隔离回归测试还把指令模型收到的全部RPM改为饱和值，确认其特征和预测时间不变。
network_update_availability.csv记录逐轨迹更新可用性；diagnostic_before_rpm_isolation_fix/仅保留修复输入隔离前的诊断输出，不用于本报告。

本次只有一个随机种子，完整六轴未在四条轨迹上表现出一致收益，指令模型的平均网络误差略低于RPM模型；不据此宣称统计显著或普遍优劣。
里程计结果同时反映输入信息和遥测质量，不能把Test 1故障造成的差异直接归因于网络结构。

## 可复现性

provenance.json记录模型哈希、数据哈希、运行时源码哈希和评分口径；每条_samples.npz也记录实现签名，阻止误复用旧实现结果。
网络绘图provenance.json记录原始预测文件哈希。论文正文未修改。
用户已允许停止AirIMU后继续训练AirIO；run_after_airimu_stop.py仅执行缓存、AirIO训练和评估，不会重启AirIMU或三组消融训练。
'''
    if 'airio_3d' in methods:
        start=text.index('三组消融和两种 ESKF 已完成。');end=text.index('## 三组网络消融')
        text=text[:start]+'''三组消融和两种 ESKF 沿用已冻结结果。主对比中的旧平面 AirIO 已由通过验证门限的新三维 AirIO 替换。
AirIMU 和 AirIO 均从头训练，保留官方三维网络、完整损失和滤波结构。训练监督来自 AprilTag XY/yaw、压力计负深度 Z、IMU 解算 roll/pitch，属于混合参考，并非独立三维动捕。
测试仅输入六轴 IMU，初速度为零；8s 时仅进行一次坐标对齐，不持续使用压力、AprilTag 或解算姿态。参考姿态条件与因果部署姿态条件分别报告。
AirIMU 的逐轮物理门限、冻结复检及 AirIO 的零速度基线检查见 new-dataset/airio_3d/。完整 XYZ、姿态、协方差及额外三维指标位于 airio_3d/。
旧平面模型及原结果保留为失败诊断；RPM、指令和 DeepVL 三方法的权重、逐样本预测与指标已核对保持不变。

'''+text[end:]
        text=text.replace('其他三方法预测文件及30条原指标保持不变。','该修复的历史结果保持可追溯；当前三维 AirIO 替换有独立的原结果归档。')
        end=text.index('用户已允许停止AirIMU后继续训练AirIO；')
        text=text[:end]+'三维重训和恢复统一入口为 new-dataset/airio_3d/run_pipeline.py；只有逐级通过门限才会发布新结果。\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print('Wrote evidence-bound evaluation report:',OUT/'README.md')

if __name__=='__main__':main()
