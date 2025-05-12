from dx_modelzoo.dataset import DatasetBase
from dx_modelzoo.enums import DatasetType, EvaluationType, SessionType
from dx_modelzoo.evaluator import EvaluatorBase
from dx_modelzoo.models import ModelBase
from dx_modelzoo.session import SessionBase

from .dicts import DATASET_DICT, EVAL_DICT, MODEL_DICT


class ModelFactory:
    """Model Factory class.
    it makes model instance from mdoel_name, session_type, model_path, and data_dir.
    """

    def __init__(self, model_name: str, session_type: SessionType, model_path: str, data_dir: str) -> None:
        self.model_name = model_name
        self.session_type = session_type
        self.model_path = model_path
        self.data_dir = data_dir

    def _get_model_cls(self) -> ModelBase:
        """get model class.

        Raises:
            ValueError: if model name is invalid, raise ValueError.

        Returns:
            ModelBase: model class.
        """
        if self.model_name in MODEL_DICT:
            return MODEL_DICT[self.model_name]
        else:
            raise ValueError(f"Invalid Model Name. {self.model_name}")

    def _get_session(self) -> SessionBase:
        """get runtime session.

        Raises:
            ValueError: if session type is invalid, raise ValueError.

        Returns:
            SessionBase: runtime session.
        """
        if self.session_type == SessionType.onnxruntime:
            from dx_modelzoo.session.onnx_runtime_session import OnnxRuntimeSession

            print("OnnxRuntime Session is created.")
            return OnnxRuntimeSession(self.model_path)
        elif self.session_type == SessionType.dxruntime:
            from dx_modelzoo.session.dx_runtime_session import DxRuntimeSession

            print("DxRuntime Session is created.")
            return DxRuntimeSession(self.model_path)
        else:
            raise ValueError(f"Invalid session type. {self.session_type}")

    def _get_dataset(self, dataset_type: DatasetType) -> DatasetBase:
        """get datset.

        Args:
            dataset_type (DatasetType): dataset type.

        Raises:
            ValueError: if dataset type is invalid, raise ValueError.

        Returns:
            DatasetBase: dataset.
        """
        if dataset_type in DATASET_DICT:
            return DATASET_DICT[dataset_type](self.data_dir)
        else:
            raise ValueError(f"Invalid Dataset Type. {dataset_type}")

    def _get_evaluator(
        self, evaluator_type: EvaluationType, session: SessionBase, dataset: DatasetBase
    ) -> EvaluatorBase:
        """get evaluator.

        Args:
            evaluator_type (EvaluationType): evaluator type.
            session (SessionBase): runtime session.
            dataset (DatasetBase): dataset.

        Raises:
            ValueError: if evaluator type is invalid, raise ValueError.

        Returns:
            EvaluatorBase: evaluator.
        """
        if evaluator_type in EVAL_DICT:
            return EVAL_DICT[evaluator_type](session, dataset)
        else:
            raise ValueError(f"Invalid Model Name. {self.model_name}")

    def make_model(self) -> ModelBase:
        """make model.

        Returns:
            ModelBase: model instance.
        """
        model_cls = self._get_model_cls()
        session = self._get_session()
        dataset = self._get_dataset(model_cls.info.dataset)
        evaluator = self._get_evaluator(model_cls.info.evaluation, session, dataset)
        return model_cls(evaluator)
