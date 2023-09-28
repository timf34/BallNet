"""
This script is used to compile the model for deployment on Nvidia Jetson Nano
"""

import boto3
import onnx
import onnxruntime as ort
import sagemaker
import torch

from network.footandball import model_factory


def load_model() -> torch.nn.Module:
    return model_factory("fb1", 'detect', ball_threshold=0.7)


def convert_torch_to_onnx():
    """
    Convert our PyTorch model to ONNX format
    """
    model = load_model()
    model.load_state_dict(torch.load("models/model_12_06_2022_2349_final_with_augs.pth"))
    model.eval()
    model = model.cuda()

    dummy_input = torch.randn(1, 3, 1920, 1080).cuda()
    onnx_model_path = "models/weights.onnx"
    model_output = model(dummy_input)

    # Export the model
    # torch.onnx.export(model, dummy_input, onnx_model_path)
    torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=12,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "output": {0: "batch_size"}},
                      keep_initializers_as_inputs=True,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                      verbose=False)


def load_onnx_model():
    return onnx.load("models/weights.onnx")


def check_onnx_model(print_graph: bool = False, extra_check: bool = False) -> None:
    onnx_model = load_onnx_model()
    onnx.checker.check_model(onnx_model)
    if print_graph:
        print(onnx.helper.printable_graph(onnx_model.graph))

    if extra_check:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4

        try:
            sess = ort.InferenceSession('models/weights.onnx', sess_options)
        except Exception as e:
            print("Error loading the model: " + str(e))
            if 'StatusCode.NOT_IMPLEMENTED' in str(e):
                print("Unsupported operators: ", sess_options.providers)


def run_inference_onnx():
    dummy_input = torch.randn(1, 3, 1920, 1080).cuda()

    # Use ort to run the model
    ort_session = ort.InferenceSession("models/weights.onnx")
    ort_outputs = ort_session.run(None, {"input": dummy_input.cpu().numpy()})
    print(ort_outputs)


def compile_onnx_model():
    pass


def main():
    convert_torch_to_onnx()
    check_onnx_model(print_graph=False)
    # run_inference_onnx()


if __name__ == '__main__':
    main()
