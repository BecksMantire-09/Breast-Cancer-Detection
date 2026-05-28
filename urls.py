from django.urls import path
from .views import predict_breast_cancer, predict_image

urlpatterns = [
     path("predict/", predict_breast_cancer, name="predict_breast_cancer"),
]