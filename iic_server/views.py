from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializer import ImageUploadSerializer
from PIL import Image
from ultralytics import YOLO
import math
import face_recognition
import numpy as np

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            # Process the image
            results = self.process_image(image)
            return Response(results, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def process_image(self, image):
        # Convert PIL image to tensor
        img = Image.open(image)
        img = img.convert("RGB")

        # Run YOLOv8 inference
        results = self.detect_product(img)
        face_name = self.face_recog(img)
        return results, face_name

    def detect_product(self, image):
        model = YOLO("./iic_api/yolov-model/best.pt")
        # Predict with the model
        results = model(image)  # predict on an image
        # for result in results:
        #     print(result["name"], result["confidence"], result["box"])
        names = model.names
        for r in results:
            boxes = r.boxes
            for box in boxes:
                return names[int(box.cls)]

    def face_recog(self, image):
        # Load a sample picture and learn how to recognize it.
        obama_image = face_recognition.load_image_file("iic_server/dataset-faces/obama.jpg")
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        # Load a second sample picture and learn how to recognize it.
        biden_image = face_recognition.load_image_file("iic_server/dataset-faces/biden.jpg")
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

        # # Test
        # unknown_image = face_recognition.load_image_file("doyoon.jpg")
        # unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        # Create arrays of known face encodings and their names
        known_face_encodings = [
            obama_face_encoding,
            biden_face_encoding,
        ]
        known_face_names = [
            "Barack Obama",
            "Joe Biden"
        ]

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        # Test (Put the received image data by raspberry pi here)

        #unknown_image = face_recognition.load_image_file(image)
        unknown_image = np.array(image)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        face_names = []
        for face_encoding in unknown_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
        return face_names