from typing import Dict, Union,List, NamedTuple, Optional, Set
import os
import requests
from collections import namedtuple
from dataclasses import asdict, dataclass
from tqdm import tqdm

from dx_modelzoo.factory.dicts.model_dict import MODEL_DICT
from dx_modelzoo.factory.model_factory import ModelFactory
from dx_modelzoo.enums import SessionType, DatasetType
from dx_modelzoo.tools.benchmark_config import Modelzoo_config
from enum import Enum

cfg = Modelzoo_config()
from loguru import logger

MODEL_LOCAL_PATH = cfg.MODEL_LOCAL_PATH
COMPILER_VER = cfg.COMPILER_VER

class APIMixin:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _make_url(self, sub_url: List[str] = None):
        if not isinstance(sub_url, list):
            print("Sub-URL need to be list of strings")
            return ""
        return os.path.join(self.base_url, *sub_url)

    def check_server(self):
        try:
            target_url = self._make_url(["meta"])
            response = requests.get(url=target_url)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            return False

    def get(self, sub_url: List[str] = [], headers: str = None, params: Dict[str, str] = None, verify: bool = True):
        target_url = self._make_url(sub_url=sub_url)
        response = requests.get(url=target_url, headers=headers, params=params, verify=verify)
        return response

    def post(self, data, headers: Dict[str, str] = None, sub_url: List[str] = []):
        target_url = self._make_url(sub_url=sub_url)
        if "application/json" in headers["Content-Type"]:
            response = requests.post(url=target_url, headers=headers, json=data)
        elif "multipart/form-data" in headers["Content-Type"]:
            response = requests.post(url=target_url, headers=headers, data=data, stream=True)
        else:
            raise ValueError(f'There is no defined request for {headers["Content-Type"]}')
        return response

    def patch(self, data, headers: Dict[str, str] = None, sub_url: List[str] = [], params: Dict[str, str] = None):
        target_url = self._make_url(sub_url=sub_url)
        response = requests.patch(url=target_url, headers=headers, json=data, params=params)
        return response

    def delete(self, params: Dict[str, str], sub_url: List[str] = []):
        target_url = self._make_url(sub_url=sub_url)
        response = requests.delete(url=target_url, params=params)
        return response
    

class FileMixin:
    supported_extension = ["json", "onnx"]

    def _generate_filename(self, filename: str):
        import uuid

        name, extension = filename.rsplit(".", maxsplit=1)
        random_hash = str(uuid.uuid4()).split("-")[0]
        return f"{name}_{random_hash}.{extension}"

    def make_path(self, path: str = None, target_file_name: str = None):
        return os.path.join(os.path.dirname(path), target_file_name)

    def get_name(self, path: str = None):
        return os.path.basename(path)

    def get_size(self, path: str = None):
        if not os.path.isfile(path=path):
            raise ValueError(f"{path} does not exists")

        return os.path.getsize(path)

    def read(self, path: str = None):
        if not os.path.exists(path=path):
            raise ValueError(f"{path} does not exists")

        # Read file
        with open(path, "rb") as file:
            return file.read()

    def save(self, content: object, path: str = None):
        import pathlib

        target_filename = self.get_name(path)

        # Check and create target dir
        target_dir = os.path.dirname(path)
        pathlib.Path(target_dir).mkdir(mode=755, parents=True, exist_ok=True)

        # Check file exists
        if pathlib.Path(path).exists():
            print(f"{target_filename} already exists.")
            path = os.path.join(target_dir, self._generate_filename(filename=target_filename))

        # Create file
        with open(path, "wb") as file:
            file.write(content)

        print(f"{path} is saved successfully.")

        return path
    

class URL(str, Enum):
    OPS = "http://192.168.0.82:8002/modelzoo/api"
    DEV = "http://192.168.0.14:8002/modelzoo/api"


class APIRequest(APIMixin, FileMixin):
    """A class for handling API requests related to database and file operations.
    Attributes:
        db_related (list): List of strings representing database-related endpoints.
        file_related (list): List of strings representing file-related endpoints.
    Methods:
        read_db(query_params): Reads data from the database based on provided query parameters.
        create_db(target, name, data): Creates data in the database for meta & performance.
        create_all(data, paths): Creates model with meta, performances, files in one.
        update_db_meta(name, data): Update model meta data.
        update_db_performance(name, data): Updates performance data in the database for a specified name.
        delete_all(name): Deletes all data related to a specified name from the database.
        download_file(name, path): Download a file associated with a specified name from the server.
        upload_file(name, path): Upload a file to the server associated with a specified name.
    """

    db_related = ["meta", "performances"]
    file_related = ["files"]

    def __init__(self, base_url: str = URL.OPS):
        super().__init__(base_url)

    def read_db(self, query_params: Dict[str, Union[str, bool, Dict[str, str]]] = {}):
        """Reads data from the database based on provided query parameters.
        If query_params is None then will read all the data registered
        Args:
            query_params (dict, optional): Dictionary containing query parameters for filtering data.
        Raises:
            ClientException: If the server is unreachable,
                          If query_params is not a dictionary.
        Returns:
            Response: Response object containing data fetched from the database.
        Usage::
            >> from dx_modelzoo.api import APIRequest
            >> from dx_modelzoo.api.utils import QuerySchema
            >>
            >> api_class = APIRequest()
            >> query_params = QuerySchema(test_mode__contains="ACCURACY").parse_data()
            >>
            >> response = api_class.read_db(query_params=query_params)
            >>
            >> content = response.json()
        """
        if not self.check_server():
            raise ValueError("Currently, server can't communicate. plz contact devops admin")

        if not isinstance(query_params, dict):
            raise ValueError("query_params should be dict type")

        sub_url = [self.db_related[0]]
        try:
            response = self.get(sub_url=sub_url, params=query_params, verify=False)
            response.raise_for_status()
        except requests.HTTPError as E:
            raise ValueError(E)

        return response

@dataclass
class QuerySchema:
    name: Optional[str] = None
    name__contains: Optional[str] = None
    task: Optional[str] = None
    task__contains: Optional[str] = None
    test_mode: Optional[str] = None
    test_mode__contains: Optional[str] = None
    dataset: Optional[str] = None
    dataset__contains: Optional[str] = None
    authorized_by: Optional[str] = None
    created_by: Optional[str] = None
    public_name: Optional[str] = None
    public_name__contains: Optional[str] = None
    compilability: Optional[str] = None
    tag: Optional[str] = None
    tag__contains: Optional[str] = None
    supported_device__contains: Optional[str] = None
    created_date__contains: Optional[str] = None

    def parse_data(self):
        data = asdict(self)
        return {key: val for key, val in data.items() if val}
    

def search_key_name_from_public_name(public_name: str) -> str:
    model_meta = download_model_meta(model_name=public_name, from_public_name=True)
    key_name = model_meta["name"]
    return key_name


def download_model_meta(model_name: str, from_public_name=False) -> dict:
    """download model meta from DB
    Args:
        model_name (str): model name
    Returns:
        dict: model meta
    """
    model_name = model_name.replace("_qmaster", "")

    if from_public_name:
        api_class, query_params = APIRequest(base_url=URL.OPS), QuerySchema(public_name=model_name).parse_data()
    else:
        api_class, query_params = APIRequest(base_url=URL.OPS), QuerySchema(name=model_name).parse_data()

    try:
        response = api_class.read_db(query_params=query_params)
        response.raise_for_status()
    except requests.HTTPError as e:
        raise ValueError(f"model_name {model_name} is invalid for api model download (error:{e})")
    else:
        model_meta = response.json()["requested_data"]
    return model_meta[0]


def parse_model_meta(model_meta: dict) -> NamedTuple:
    name_string = " ".join(list(model_meta.keys()))
    ModelMeta = namedtuple("ModelMeta", name_string)
    model_meta = ModelMeta(**model_meta)
    return model_meta

def evaluate_models(model_list_by_task: Dict[str, List[str]],
                    model_ignore_list: Set[str],
                    data_dir_dict: Dict[str, str]
                    ) -> None:
    """
    Evaluate models across different tasks while ignoring specified models.
    
    Args:
        model_list_by_task: Dictionary mapping tasks to model names
        model_ignore_list: Set of model names to skip
        data_dir_dict: Dictionary mapping datasets to data directories
    """
    for task_name, model_names in model_list_by_task.items():
        for model_index, model_name in enumerate(model_names):
            logger.info(f"@JSON <START Evaluation> [{model_name}]")
            if model_name in model_ignore_list:
                logger.warning(f"Skipping ignored or unregistered No.{model_index+1} model: {model_name}")
                continue
            try:
                logger.debug(f"Initializing {task_name} No.{model_index+1} model: {model_name}| [{model_index+1}/{len(model_names)}]")
                model_path = make_model_path_S3(model_name)
                data_dir = data_dir_dict[MODEL_DICT[model_name].info.dataset]
                model =  ModelFactory(model_name, SessionType.dxruntime,model_path, data_dir).make_model()
                logger.debug(f"@JSON [{task_name} : {model_name}] Model initialization completed>")
                model.eval()
                logger.success(f"@JSON [{task_name} : {model_name}] <Evaluation COMPLETED>")
                del model     
            except Exception as e:
                logger.opt(exception=True).error(f"Runtime error in {model_name} : {e}")

              
def evaluate_models_internal(model_names: List[str],
                    local_model_list_path: str,
                    model_list_by_task: Dict[str, List[str]],
                    data_dir_dict: Dict[str, str]
                    ) -> None:
    model_registered_list = set()
    for models in model_list_by_task.values():
        model_registered_list.update(models)
    
    for model_name in tqdm(model_names, desc="Evaluating models on NPU"):
        logger.info(f"@JSON <START Evaluation> [{model_name}]")
        if model_name not in model_registered_list:
            logger.warning(f"@JSON *** Error code = -2 *** - Skipping unregistered Model : [{model_name}].")
            continue
        
        task_name = None
        for task, models in model_list_by_task.items():
            if model_name in models:
                task_name = task
                break
        
        if not task_name:
            logger.warning(f"@JSON *** Error code = -2 *** - [{model_name}] not found in tasks.")
            continue
                 
        try:
            model_path, name_parts = make_model_path_local(local_model_list_path, model_name)
            if not os.path.isfile(model_path):
                raise FileNotFoundError()
                
            data_dir = data_dir_dict[MODEL_DICT[model_name].info.dataset]
            model =  ModelFactory(model_name, SessionType.dxruntime,model_path, data_dir).make_model()
            logger.debug(f"@JSON [{task_name} : {model_name}] Model initialization completed>")
            model.eval()
            logger.success(f"@JSON [{task_name} : {model_name}] <Evaluation COMPLETED>")
            del model 
        except FileNotFoundError as e:
           logger.warning(f"@JSON *** Error code = -3 *** - Model file not found [{task_name} : {model_name}] at {model_path} {e}.")
        except Exception as e:
            logger.error(f"@JSON *** Error code = -1 *** - Inference failure [{task_name} : {model_name}]: {e}.")  # RuntimeError
        finally:
            pass
                
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    filename = os.path.basename(save_path)
    print(f"{filename} Model downloading from Deepx SDK Server...")
    print(f"{filename} Model size: {total_size / (1024 * 1024):.2f} MB")
    
    with open(save_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    print(f"Model saved to: {save_path}")
    
# # local folder for debug 
# dir_path = "/home/devops/ws_modelzoo/open_model_compiled/compiled_1203"

def make_model_path_S3(model_name):
    public_name = search_key_name_from_public_name(model_name)
    # make folder for models
    os.makedirs(f"{MODEL_LOCAL_PATH}/dxnn/{COMPILER_VER}/", exist_ok=True)
    # download models from amazon server.
    url = f"https://sdk.deepx.ai/modelzoo/dxnn/{COMPILER_VER}/{public_name}.dxnn"
    model_local_path = f"{MODEL_LOCAL_PATH}/dxnn/{COMPILER_VER}/{public_name}.dxnn"
    download_file(url, model_local_path)
    return model_local_path

def make_model_path_local(local_model_list_path, model_name):
    folder_name_list = os.listdir(local_model_list_path)
    for folder_name in folder_name_list:
        name_parts = extract_folder_name(folder_name)
        if model_name == name_parts[0]:
            model_local_path = f"{local_model_list_path}/{folder_name}/{model_name}.dxnn"
            return model_local_path, name_parts
    return None

def extract_folder_name(folder_name):
    parts = folder_name.split('-')
    if len(parts) == 3: 
        first_part, second_part, third_part = parts
        return (first_part, second_part, third_part)

api = APIRequest()

# if __name__=="__main__":
#     data_dir_dict = {
#     DatasetType.imagenet: "/mnt/datasets/ILSVRC2012/val",
#     DatasetType.coco: "/mnt/datasets/COCO/official",
#     DatasetType.voc_od: "/mnt/datasets/PascalVOC/VOCdevkit/VOC2007",
#     DatasetType.bsd68: "/mnt/datasets/BSD68",
#     DatasetType.city: "/mnt/datasets/cityscapes",
#     DatasetType.voc_seg: "/mnt/datasets/PascalVOC/VOCdevkit/VOC2012",
#     DatasetType.widerface: "/mnt/datasets/widerface"
# }
#     model_list_by_task = \
#     {'test':['BiSeNetV1'],
#     'image_classfication':['ResNet18','ResNet34','ResNet50', 'ResNet101', 'ResNet152', 'AlexNet', 'VGG11', 'VGG13', 'VGG13BN', 'VGG11BN', 'VGG19BN', 'MobileNetV1', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large','SqueezeNet1_0', 'SqueezeNet1_1', 'EfficientNetB2', 'EfficientNetV2S', 'RegNetX400MF', 'RegNetX800MF', 'RegNetY200MF', 'RegNetY400MF', 'RegNetY800MF', 'ResNeXt50_32x4d', 'ResNeXt26_32x4d', 'DenseNet121', 'DenseNet161','HarDNet39DS', 'WideResNet50_2', 'WideResNet101_2', 'OSNet0_25', 'OSNet0_5', 'RepVGGA1'],
#     'object_detection':['YoloV3', 'YoloV5N', 'YoloV5S', 'YoloV5M', 'YoloV5L', 'YoloV7', 'YoloV7E6', 'YoloV7Tiny', 'YoloV8L', 'YoloV9C', 'YoloV9S', 'YoloV9T', 'YoloXS','SSDMV1', 'SSDMV2Lite'],
#     'face_id':['YOLOv5s_Face', 'YOLOv5m_Face', 'YOLOv7s_Face', 'YOLOv7_Face', 'YOLOv7_TTA_Face', 'YOLOv7_W6_Face', 'YOLOv7_W6_TTA_Face'],
#     'semantic_segmentation':['DeepLabV3PlusMobilenet',  'DeepLabV3PlusDRN', 'DeepLabV3PlusMobileNetV2', 'DeepLabV3PlusResNet101', 'DeepLabV3PlusResNet50', 'DeepLabV3PlusResnet', 'BiSeNetV1', 'BiSeNetV2',],
#     'image_denoising':['DnCNN_15', 'DnCNN_25', 'DnCNN_50']}

#     model_ignore_list = ['','DeepLabV3PlusDRN', 'DeepLabV3PlusMobileNetV2','DeepLabV3PlusResNet101',
#                         'DeepLabV3PlusResNet50','DeepLabV3PlusResnet', 'MobileNetV3Small' ]

#     evaluate_models(
#         # model_list_by_task={k: v for k, v in model_list_by_task.items() if k == 'face_id'},
#         model_list_by_task=model_list_by_task,
#         model_ignore_list=model_ignore_list,
#         data_dir_dict=data_dir_dict
#     ) 
    
#     # get_public_name = False
#     # if get_public_name:    
#     #     public_name_list =[]
#     #     for task in model_list_by_task:
#     #         for model in model_list_by_task[task]:
                
#     #             public_name = _test_make_model_path(model)
#     #             print(f"{model} public name : {public_name}")
#     #             public_name_list.append(f"{public_name}")
#     #     print(public_name_list)