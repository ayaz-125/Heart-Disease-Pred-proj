import mlflow.pyfunc
import joblib
import os

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.scaler = joblib.load(context.artifacts["scaler"])

    def predict(self, context, model_input):
        scaled_input = self.scaler.transform(model_input)
        return self.model.predict(scaled_input)
