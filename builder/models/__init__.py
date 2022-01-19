import importlib
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_detector_model(args):

    model_module = importlib.import_module("builder.models.detector_models." + args.model)
    model = getattr(model_module, args.model.upper())

    return model


def grad_cam(args, model, data):
    target_layer = model.classifier[-1]
    input_tensor = data # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    use_cuda = not args.cpu
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = args.output_dim

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
