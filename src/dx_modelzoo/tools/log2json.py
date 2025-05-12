import re
import json
import os
from collections import defaultdict
import subprocess
from datetime import date
import torch
from typing import Dict, DefaultDict, Optional, Tuple
from dx_modelzoo.tools.benchmark_config import Modelzoo_config

class Run_benchmark_config(Modelzoo_config):
    def __init__(self):
        super().__init__()
        self.TASK_MAPPING = {
            "image_classfication": "ImageClassification",
            "object_detection": "ObjectDetection",
            "face_id": "FaceID",
            "semantic_segmentation": "SemanticSegmentation",
            "image_denoising": "ImageDenoising"
        }

        self.ERROR_CODE_MAPPING = {
            -1: "DeepX Runtime Error", -2: "Unregistered Model",
            -3: "File Not Found", -9: "Unknown Error"
        }

        self.DEFAULT_VERSIONS = {
            "dxrt_version": "2.6.0", "rt_driver_version": "1.3.0",
            "fw_version": "unknown", "compiler_version": "unknown"
        }


def parse_log(file_path: str) -> DefaultDict:
    if isinstance(file_path, tuple):
        file_path = file_path[-1]
    cfg = Run_benchmark_config()
    runtime_type = "DXRT" if "dxrt" in os.path.basename(file_path) else "ONNXRT"
    results = defaultdict(lambda: defaultdict(dict))
    current_task = None
    current_model = None
    
    with open(file_path, 'r') as f:
        for line in f:           
            # 提取每个模型名称
            model_match = re.search(r'@JSON <START Evaluation>\s*\[([^\]]+)\]', line)
            if model_match:
                current_model = model_match.group(1).strip()
                continue
            
            # 检查此模型属于什么任务，如果模型没有注册。任务为“Unknown”
            current_task = "Unknown"
            for task, models in cfg.model_list_by_task.items():
                if current_model in models:
                    current_task = task
                    break
            
            # 错误检查
            if "*** Error code =" in line:
                error_match = re.search(r'\*\*\* Error code = (-\d+) \*\*\*', line)
                if error_match and current_model and current_task:
                    error_code = int(error_match.group(1))
                    results[current_task][current_model][runtime_type] = {
                        "error_code": error_code,
                        "error_message": cfg.ERROR_CODE_MAPPING.get(error_code, f"Unknown Error Code: {error_code}")
                    }
                    current_model = None
                    current_task = None
                    error_match = None
                    continue
            
            if "initialization completed" in line:
                continue
                
            # 根据任务类型解析指标
            if current_model and current_task:
                # 图像分类指标
                if "Top1 Accuracy:" in line and "Top5 Accuracy:" in line and "@JSON <" in line:
                    acc_match = re.search(r'@JSON <Top1 Accuracy:([\d.]+); Top5 Accuracy:([\d.]+); Average FPS:([\d.]+)>', line)
                    if acc_match:
                        metrics = {
                            "Top1 Accuracy": float(acc_match.group(1)),
                            "Top5 Accuracy": float(acc_match.group(2)),
                            "Average FPS": float(acc_match.group(3)),
                        }
                        results[current_task][current_model][runtime_type] = metrics
                        continue
                
                # 对象检测指标 (mAP, mAP50)
                elif "mAP:" in line and "mAP50:" in line and "@JSON <" in line:
                    map_match = re.search(r'@JSON <mAP:([\d.]+); mAP50:([\d.]+); Average FPS:([\d.]+)>', line)
                    if map_match:
                        metrics = {
                            "mAP@0.5:0.95": float(map_match.group(1)),
                            "mAP@0.5": float(map_match.group(2)),
                            "Average FPS": float(map_match.group(3)),
                        }
                        results[current_task][current_model][runtime_type] = metrics
                        continue
                # 对象检测指标 (mAP@0.5)
                elif "mAP@0.5" in line and "@JSON <" in line:
                    map_match = re.search(r'@JSON <mAP@0.5:([\d.]+); Average FPS:([\d.]+)>', line)
                    if map_match:
                        metrics = {
                            "mAP@0.5": float(map_match.group(1)),
                            "Average FPS": float(map_match.group(2)),
                        }
                        results[current_task][current_model][runtime_type] = metrics
                        continue
                
                # 语义分割指标
                elif "mIoU:" in line and "@JSON <" in line:
                    miou_match = re.search(r'@JSON <mIoU:([\d.]+); Average FPS:([\d.]+)>', line)
                    if miou_match:
                        metrics = {
                            "mIoU": float(miou_match.group(1)),
                            "Average FPS": float(miou_match.group(2)),
                        }
                        results[current_task][current_model][runtime_type] = metrics
                        continue
                
                # 图像降噪指标
                elif "Average PSNR:" in line and "Average SSIM:" in line and "@JSON <" in line:
                    psnr_match = re.search(r'@JSON <Average PSNR:([\d.]+); Average SSIM:([\d.]+); Average FPS:([\d.]+)>', line)
                    if psnr_match:
                        metrics = {
                            "Average PSNR": float(psnr_match.group(1)),
                            "Average SSIM": float(psnr_match.group(2)),
                            "Average FPS": float(psnr_match.group(3)),
                        }
                        results[current_task][current_model][runtime_type] = metrics
                        continue
                
                # 人脸识别指标
                elif "@JSON <Easy Val AP:" in line:
                    face_match = re.search(r'@JSON <Easy Val AP:([\d.]+); Medium Val AP:([\d.]+); Hard Val AP:([\d.]+); Average FPS:([\d.]+)>', line)
                    if face_match:
                        metrics = {
                            "Easy Val AP": float(face_match.group(1)),
                            "Medium Val AP": float(face_match.group(2)),
                            "Hard Val AP": float(face_match.group(3)),
                            "Average FPS": float(face_match.group(4)),
                        }
                        results[current_task][current_model][runtime_type] = metrics
                        continue
            
            # 标记评估完成
            if "<Evaluation COMPLETED>" in line and current_model and current_task:
                if current_task in results and current_model in results[current_task]:
                    if runtime_type in results[current_task][current_model]:
                        results[current_task][current_model][runtime_type]["completed"] = True
    return results


def extract_version_info(dxnn_path: str) -> Dict[str, str]:
    """从系统和DXNN文件中提取版本信息"""
    cfg = Run_benchmark_config()
    versions = cfg.DEFAULT_VERSIONS.copy()
    
    # 从DXNN提取编译器版本
    try:
        with open(dxnn_path, 'rb') as f:
            bs = f.read()
            header_dict = json.loads(bs[8:8192].decode().rstrip("\x00"))
            content = bs[8192:]
            
            data = header_dict.get("data", {})
            compile_config = data.get("compile_config", {})
            
            if "offset" in compile_config and "size" in compile_config:
                offset, size = compile_config["offset"], compile_config["size"]
                config_json = content[offset:offset+size].decode()
                config = json.loads(config_json)
                versions["compiler_version"] = config.get("compile_version", "unknown")
    except Exception as e:
        print(f"Error reading DXNN file: {e}")
    
    try:
        result = subprocess.run(['dxrt-cli', '-s'], capture_output=True, text=True)
        output = result.stdout
        
        dxrt_match = re.search(r'DXRT v([\d.]+)', output)
        if dxrt_match:
            versions["dxrt_version"] = dxrt_match.group(1)
        
        driver_match = re.search(r'\* RT Driver version   : v([\d.]+)', output)
        if driver_match:
            versions["rt_driver_version"] = driver_match.group(1)
        
        fw_match = re.search(r'\* FW version          : v([\d.]+)', output)
        if fw_match:
            versions["fw_version"] = fw_match.group(1)
    except Exception as e:
        print(f"Error running dxrt-cli: {e}")
    return versions


def merge_results(dxrt_results: DefaultDict, onnxrt_results: DefaultDict) -> Dict:
    """合并DXRT和ONNXRT的结果"""
    merged = {}
    all_tasks = set(dxrt_results.keys()) | set(onnxrt_results.keys())
    
    for task in all_tasks:
        task_models = {}
        all_models = set(dxrt_results.get(task, {}).keys()) | set(onnxrt_results.get(task, {}).keys())
        
        for model in all_models:
            model_data = {}
            
            # 如果有DXRT数据则添加
            if task in dxrt_results and model in dxrt_results[task] and "DXRT" in dxrt_results[task][model]:
                model_data["DXRT"] = dxrt_results[task][model]["DXRT"]
            
            # 如果有ONNXRT数据则添加
            if task in onnxrt_results and model in onnxrt_results[task] and "ONNXRT" in onnxrt_results[task][model]:
                model_data["ONNXRT"] = onnxrt_results[task][model]["ONNXRT"]
            
            # 只有当有数据时才添加模型
            if model_data:
                task_models[model] = model_data
        
        # 只有当有模型时才添加任务
        if task_models:
            merged[task] = task_models
    
    return merged


def build_metadata(versions: Dict[str, str]) -> Dict[str, str]:
    """构建包含版本和系统信息的元数据"""
    metadata = {
        "Experiment Date": date.today().isoformat(),
        "FW Version": versions["fw_version"],
        "DX Runtime Version": versions["dxrt_version"],
        "DX Driver Version": versions["rt_driver_version"],
        "DX Compiler Version": versions["compiler_version"],
    }
    
    # 如果有GPU则添加GPU信息
    if torch.cuda.is_available():
        metadata.update({
            "GPU Model": torch.cuda.get_device_name(0),
            "CUDA Version": torch.version.cuda
        })
    
    return metadata


def export_benchmark_report(
    dxrt_log: str, 
    onnxrt_log: str, 
    sample_dxnn_path: str,
    output_path: str,
    ) -> None:
    if isinstance(output_path, tuple):
        output_path = output_path[-1]
    
    dxrt_results = defaultdict(lambda: defaultdict(dict))
    onnxrt_results = defaultdict(lambda: defaultdict(dict))
    
    # log parse
    if dxrt_log is not None:
        dxrt_results = parse_log(dxrt_log)
    
    if onnxrt_log is not None:
        onnxrt_results = parse_log(onnxrt_log)
    
    merged_results = merge_results(dxrt_results, onnxrt_results)
    
    # get versions
    versions = extract_version_info(sample_dxnn_path)
    
    # Get metadata
    metadata = build_metadata(versions)
    
    # Get final output 
    final_output = {
        "metadata": metadata,
        "results": merged_results
    }
    
    # save results to json
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"Model zoo benchmarking Report is saved at {output_path}")


# Debug
if __name__ == "__main__":
    dxrt_log='./log_dxrt_eval_internal.log'
    onnxrt_log='./log_onnxrt_eval_internal.log'
    
    if "internal" in dxrt_log:
        export_benchmark_report(
            dxrt_log=dxrt_log,
            onnxrt_log=onnxrt_log,
            sample_dxnn_path="/mnt/regression_storage/accuracy_modelzoo_data/M1A/2/ResNet101-ResNet101-1/ResNet101.dxnn",
            output_path='benchmark_report_internal.json',
        )
    else:
        export_benchmark_report(
            dxrt_log=dxrt_log,
            onnxrt_log=onnxrt_log,
            sample_dxnn_path="./open_models/dxnn/1_32_4/ResNet18-1.dxnn",
            output_path='benchmark_report.json',
        )