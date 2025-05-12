from prefect import task, flow, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

import time, argparse
from dx_modelzoo.tools.DXRT_eval import evaluate_models as dxrt_eval
from dx_modelzoo.tools.ONNXRT_eval import evaluate_models as onnxrt_eval
from dx_modelzoo.tools.log2json import export_benchmark_report
from dx_modelzoo.tools.benchmark_config import Modelzoo_config

from loguru import logger

class Run_benchmark_config(Modelzoo_config):
    def __init__(self, args=None):
        super().__init__(args)
        
        # DEBUG: hijack model list
        if args.debug:
            self.model_list_by_task = {
                'image_classfication': [
                    'ResNet18'
                ],
                'object_detection': [
                    'YoloV3',
                ],
                'face_id': [
                    'YOLOv5s_Face',
                ],
                'semantic_segmentation': [
                    'DeepLabV3PlusMobilenet',
                ],
                'image_denoising': ['DnCNN_15',],
            }

        # self.model_ignore_list = []
        self.model_ignore_list = [
            '', 'DeepLabV3PlusDRN', 'DeepLabV3PlusMobileNetV2', 'DeepLabV3PlusResNet101',
            'DeepLabV3PlusResNet50', 'DeepLabV3PlusResnet', 'MobileNetV3Small', 'ResNet152'
        ]
        self.dxrt_log='./log_dxrt_eval.log',
        self.onnxrt_log='./log_onnxrt_eval.log',
        self.sample_dxnn_path="./open_models/dxnn/1_32_4/ResNet18-1.dxnn",
        self.output_path='benchmark_report.json',

        
@task
def setup_logging(device_type):
    """Setup logging for the evaluation task."""
    log_id = logger.add(
        f"log_{device_type.lower()}_eval.log",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            f"<magenta>{device_type}</magenta> | "
            "<level>{level: <8}</level> | "
            "<cyan>{process.name}</cyan> | "
            "{message}"
        ),
        filter=lambda record: record.get("extra", {}).get("device") == device_type,
        rotation="500 MB",
        enqueue=True,
        catch=True
    )
    return log_id

@task
def run_evaluation(eval_func, device_type, model_list_by_task, model_ignore_list, data_dir_dict):
    """Run model evaluation with proper logging context."""
    log_id = setup_logging(device_type)
    try:
        with logger.contextualize(device=device_type):
            eval_func(
                model_list_by_task=model_list_by_task,
                model_ignore_list=model_ignore_list,
                data_dir_dict=data_dir_dict
            )
    finally:
        """Clean up logging resources."""
        logger.remove(log_id)

@flow(
    name="Model Zoo Evaluation Benchmark",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True,
    description="Run DXRT and ONNXRT evaluations in parallel"
)

def main(args):
    config = Run_benchmark_config(args)
    
    print(
        "\n"
        "############ Benchmark Message ###################\n"
        "# This script will beachmark all models on both  #\n"
        "# NPU and GPU(CPU). The process will take a few  #\n" 
        "# hours. Please monitor result updating by       #\n"
        "# 'tail -f log_ONNXRT_eval.log log_DXRT_eval.log'#\n"
        "##################################################\n"
    )
    
    # Concurrent execution of onnxrt and dxrt evaluations
    onnxrt_future = None
    dxrt_future = None
    
    if args.onnxrt or args.all:
        onnxrt_future = run_evaluation.submit(
            eval_func=onnxrt_eval,
            device_type="ONNXRT",
            model_list_by_task=config.model_list_by_task,
            model_ignore_list=config.model_ignore_list,
            data_dir_dict=config.data_dir_dict
        )
    
    # Add delay between task starts if needed
    time.sleep(2.5)
    
    if args.dxrt or args.all:
        dxrt_future = run_evaluation.submit(
            eval_func=dxrt_eval,
            device_type="DXRT",
            model_list_by_task=config.model_list_by_task,
            model_ignore_list=config.model_ignore_list,
            data_dir_dict=config.data_dir_dict
        )
    
    # Wait for both tasks to complete
    if onnxrt_future:
        onnxrt_future.result()
    if dxrt_future:
        dxrt_future.result()
    
    # Generate the final report
    dxrt_log=None
    onnxrt_log=None
    flag_generate_repot=False
    
    if args.all:
        flag_generate_repot=True
        dxrt_log=config.dxrt_log
        onnxrt_log=config.onnxrt_log
    else:
        if args.onnxrt:
            flag_generate_repot=True
            onnxrt_log=config.onnxrt_log
        
        if args.dxrt:
            flag_generate_repot=True
            dxrt_log=config.dxrt_log
    
    if flag_generate_repot:
        export_benchmark_report(
            dxrt_log=dxrt_log,
            onnxrt_log=onnxrt_log,
            sample_dxnn_path=config.sample_dxnn_path,
            output_path=config.output_path,
        )
    else:
        print(
            "\n"
            "############ Benchmark Message ###################\n"
            "# Skip to make benchmark report.                 #\n"
            "##################################################\n"
        )
    
    
    print(
        "\n"
        "############ Benchmark Message ###################\n"
        "# Model Zoo benchmark has been sucessfully       #\n"
        "# conducted. Please find ./benchmark_report.json #\n"
        "##################################################\n"
    )
