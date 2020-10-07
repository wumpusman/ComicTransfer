from google.cloud import automl


prediction_client = automl.PredictionServiceClient()
project_id="hidden"
model_id="hidden"

# Get the full path of the model.
model_full_id = automl.AutoMlClient.model_path(
    project_id, "us-central1", model_id
)

file_path="../../data/005.png"
with open(file_path, "rb") as content_file:
    content = content_file.read()

image = automl.Image(image_bytes=content)
payload = automl.ExamplePayload(image=image)


params = {}

request = automl.PredictRequest(
    name=model_full_id,
    payload=payload,
    params=params
)
response = prediction_client.predict(request=request)

print("Prediction results:")
for result in response.payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))