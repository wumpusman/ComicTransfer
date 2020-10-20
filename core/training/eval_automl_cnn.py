import argparse
import os

from deprecated import deprecated


@deprecated(
    version="1.0",
    reason=
    "poor results, ml model on google servers is offline, code should be refactored,and new models need to be added"
)
def produce_automl_results(dir_img_path: str = "",
                           dir_destination_path: str = ""):
    from google.cloud import automl

    project_id = "hidden"
    model_id = "hidden"

    imgs = os.listdir(dir_img_path)
    relevant_images = [
        os.path.join(dir_img_path, i) for i in imgs if "png" in i
    ]

    prediction_client = automl.PredictionServiceClient()
    project_id = "hidden"
    model_id = "hidden"

    # Get the full path of the model.
    model_full_id = automl.AutoMlClient.model_path(project_id, "us-central1",
                                                   model_id)

    file_path = "../../data/005.png"
    with open(file_path, "rb") as content_file:
        content = content_file.read()

    image = automl.Image(image_bytes=content)
    payload = automl.ExamplePayload(image=image)

    params = {}

    request = automl.PredictRequest(name=model_full_id,
                                    payload=payload,
                                    params=params)
    response = prediction_client.predict(request=request)

    print("Prediction results:")
    for result in response.payload:
        print("Predicted class name: {}".format(result.display_name))
        print("Predicted class score: {}".format(result.classification.score))
