import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import logging
import time
import traceback

# Configure logging - simplified for production
logging.basicConfig(level=logging.INFO, 
                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                 handlers=[
                     logging.StreamHandler()  # Only log to console, no file logging
                 ])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
  static_image_mode=True,
  max_num_faces=1,
  refine_landmarks=True,
  min_detection_confidence=0.5
)

# Add a route for the root URL
@app.route('/', methods=['GET'])
def index():
  return jsonify({
      "status": "ok",
      "message": "Ray-Ban AI Sunglasses API is running",
      "endpoints": {
          "health_check": "/api/health",
          "analyze_face": "/api/analyze-face",
          "more_recommendations": "/api/more-recommendations"
      }
  })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Ray-Ban sunglasses database
def get_sunglasses_database():
  return [
      {
          "id": 1,
          "name": "Ray-Ban Aviator Classic",
          "price": 161,
          "description": "The iconic Aviator, first designed in 1937, is a timeless model that combines great aviator styling with exceptional quality.",
          "colors": ["#B1A151", "#2F2F2F", "#713F17"],
          "image": "/images/sunglasses/aviator.png",
          "recommendedFor": ["Oval", "Heart", "Square"],
      },
      {
          "id": 2,
          "name": "Ray-Ban Wayfarer",
          "price": 155,
          "description": "The iconic Ray-Ban Wayfarer is simply the most recognizable style in sunglasses. The distinct shape is paired with the traditional Ray-Ban signature logo on the sculpted temples.",
          "colors": ["#000000", "#894E3F", "#2F2F2F"],
          "image": "/images/sunglasses/wayfarer.png",
          "recommendedFor": ["Oval", "Round", "Diamond"],
      },
      {
          "id": 3,
          "name": "Ray-Ban Clubmaster",
          "price": 171,
          "description": "The Ray-Ban Clubmaster sunglasses are retro and timeless. Inspired by the 50's, the unmistakable design of the Clubmaster has graced the faces of cultural intellectuals and Hollywood icons alike.",
          "colors": ["#000000", "#713F17", "#2F2F2F"],
          "image": "/images/sunglasses/clubmaster.png",
          "recommendedFor": ["Oval", "Diamond", "Heart"],
      },
      {
          "id": 4,
          "name": "Ray-Ban Round Metal",
          "price": 161,
          "description": "Ray-Ban Round Metal sunglasses are completely round and metal, a design inspired by the counter-culture of the 1960s. This style is worn by many Hollywood celebrities and fashion icons.",
          "colors": ["#B1A151", "#2F2F2F", "#000000"],
          "image": "/images/sunglasses/round-metal.png",
          "recommendedFor": ["Square", "Heart", "Diamond"],
      },
      {
          "id": 5,
          "name": "Ray-Ban Justin Classic",
          "price": 145,
          "description": "The Ray-Ban Justin is a modern take on the iconic Wayfarer shape. With a slightly larger frame and rubberized finish, it's a contemporary look for those with an edgy style.",
          "colors": ["#000000", "#2F2F2F", "#713F17"],
          "image": "/images/sunglasses/justin.png",
          "recommendedFor": ["Round", "Oval", "Heart"],
      },
      {
          "id": 6,
          "name": "Ray-Ban Erika",
          "price": 141,
          "description": "A fresh new take on the iconic Ray-Ban Wayfarer, the Erika sunglasses feature a softer eye shape and thinner frame profile for a feminine look.",
          "colors": ["#000000", "#894E3F", "#2F2F2F"],
          "image": "/images/sunglasses/erika.png",
          "recommendedFor": ["Square", "Round", "Diamond"],
      },
      {
          "id": 7,
          "name": "Ray-Ban Hexagonal",
          "price": 154,
          "description": "The Ray-Ban Hexagonal sunglasses feature flat hexagonal lenses and thin metal temples for a distinctive geometric look.",
          "colors": ["#B1A151", "#2F2F2F", "#000000"],
          "image": "/images/sunglasses/hexagonal.png",
          "recommendedFor": ["Round", "Oval", "Heart"],
      },
      {
          "id": 8,
          "name": "Ray-Ban Caravan",
          "price": 154,
          "description": "The Ray-Ban Caravan is a squared-off version of the iconic Aviator, offering a more structured look while maintaining the classic appeal.",
          "colors": ["#B1A151", "#2F2F2F", "#000000"],
          "image": "/images/sunglasses/caravan.png",
          "recommendedFor": ["Oval", "Round", "Heart"],
      },
      {
          "id": 9,
          "name": "Ray-Ban State Street",
          "price": 163,
          "description": "The Ray-Ban State Street sunglasses feature a bold, squared shape with thick acetate frames for a statement-making look.",
          "colors": ["#000000", "#894E3F", "#2F2F2F"],
          "image": "/images/sunglasses/state-street.png",
          "recommendedFor": ["Oval", "Round", "Heart"],
      },
      {
          "id": 10,
          "name": "Ray-Ban Nomad",
          "price": 176,
          "description": "The Ray-Ban Nomad combines vintage-inspired design with modern materials, featuring a distinctive squared shape and gradient lenses.",
          "colors": ["#713F17", "#000000", "#2F2F2F"],
          "image": "/images/sunglasses/nomad.png",
          "recommendedFor": ["Oval", "Heart", "Diamond"],
      },
      {
          "id": 11,
          "name": "Ray-Ban Jack",
          "price": 161,
          "description": "The Ray-Ban Jack offers a hexagonal shape with thin metal frames for a sophisticated, intellectual look.",
          "colors": ["#B1A151", "#000000", "#2F2F2F"],
          "image": "/images/sunglasses/jack.png",
          "recommendedFor": ["Round", "Square", "Oval"],
      },
      {
          "id": 12,
          "name": "Ray-Ban Nina",
          "price": 149,
          "description": "The Ray-Ban Nina features a cat-eye silhouette with acetate frames, perfect for adding a touch of vintage glamour to any outfit.",
          "colors": ["#000000", "#894E3F", "#2F2F2F"],
          "image": "/images/sunglasses/nina.png",
          "recommendedFor": ["Heart", "Oval", "Square"],
      },
      {
          "id": 13,
          "name": "Ray-Ban Meteor",
          "price": 163,
          "description": "The Ray-Ban Meteor offers a bold, squared shape with slightly upswept corners for a distinctive retro look.",
          "colors": ["#000000", "#713F17", "#2F2F2F"],
          "image": "/images/sunglasses/meteor.png",
          "recommendedFor": ["Round", "Oval", "Diamond"],
      },
      {
          "id": 14,
          "name": "Ray-Ban Caribbean",
          "price": 168,
          "description": "The Ray-Ban Caribbean features a unique squared shape with a double bridge for a distinctive, vintage-inspired look.",
          "colors": ["#B1A151", "#000000", "#2F2F2F"],
          "image": "/images/sunglasses/caribbean.png",
          "recommendedFor": ["Oval", "Heart", "Round"],
      },
      {
          "id": 15,
          "name": "Ray-Ban Olympian",
          "price": 171,
          "description": "The Ray-Ban Olympian features a distinctive wraparound design with a single bridge for a sporty, retro-inspired look.",
          "colors": ["#000000", "#713F17", "#2F2F2F"],
          "image": "/images/sunglasses/olympian.png",
          "recommendedFor": ["Square", "Diamond", "Oval"],
      },
      {
          "id": 16,
          "name": "Ray-Ban Blaze Wayfarer",
          "price": 183,
          "description": "The Ray-Ban Blaze Wayfarer updates the classic Wayfarer with flat lenses that extend beyond the frame for a bold, contemporary look.",
          "colors": ["#000000", "#894E3F", "#2F2F2F"],
          "image": "/images/sunglasses/blaze-wayfarer.png",
          "recommendedFor": ["Oval", "Round", "Diamond"],
      },
      {
          "id": 17,
          "name": "Ray-Ban Oval",
          "price": 161,
          "description": "The Ray-Ban Oval features a distinctive oval shape with thin metal frames for a refined, intellectual look.",
          "colors": ["#B1A151", "#000000", "#2F2F2F"],
          "image": "/images/sunglasses/oval.png",
          "recommendedFor": ["Square", "Heart", "Diamond"],
      },
      {
          "id": 18,
          "name": "Ray-Ban Andrea",
          "price": 157,
          "description": "The Ray-Ban Andrea features a geometric shape with thin metal frames for a sophisticated, contemporary look.",
          "colors": ["#000000", "#B1A151", "#2F2F2F"],
          "image": "/images/sunglasses/andrea.png",
          "recommendedFor": ["Round", "Oval", "Heart"],
      },
      {
          "id": 19,
          "name": "Ray-Ban Ja-Jo",
          "price": 149,
          "description": "The Ray-Ban Ja-Jo features a perfectly round shape with thin metal frames for a minimalist, contemporary look.",
          "colors": ["#000000", "#B1A151", "#894E3F"],
          "image": "/images/sunglasses/ja-jo.png",
          "recommendedFor": ["Square", "Heart", "Diamond"],
      },
      {
          "id": 20,
          "name": "Ray-Ban Highstreet",
          "price": 163,
          "description": "The Ray-Ban Highstreet collection offers a variety of bold, fashion-forward shapes for those who want to make a statement.",
          "colors": ["#000000", "#713F17", "#2F2F2F"],
          "image": "/images/sunglasses/highstreet.png",
          "recommendedFor": ["Oval", "Round", "Heart"],
      },
  ]

# Helper function to get explanation for face shape
def get_explanation_for_face_shape(face_shape):
  explanations = {
      "Oval": "Your oval face shape is versatile and balanced. Most frame styles work well with your proportions, but rectangular and geometric frames particularly complement your natural balance.",
      "Round": "Your round face has soft curves and full cheeks. Angular and rectangular frames create contrast and help elongate your face, adding definition to your features.",
      "Square": "Your square face has a strong jawline and forehead. Round or oval frames soften your angular features and create a pleasing contrast with your face shape.",
      "Heart": "Your heart-shaped face is wider at the forehead and narrower at the chin. Frames that are wider at the bottom balance your proportions and complement your features.",
      "Diamond": "Your diamond face has narrow forehead and jawline with wider cheekbones. Oval and rimless frames highlight your cheekbones while softening your angular features.",
      "Oblong": "Your oblong face is longer than it is wide. Frames with more depth than width help create the illusion of a shorter, more balanced face.",
      "Triangle": "Your triangle face has a narrow forehead and wider jawline. Frames that are wider at the top add balance to your face shape.",
  }
  
  return explanations.get(face_shape, "Your face shape works well with these frames, creating a balanced and flattering look.")

# Function to get sunglasses recommendations based on face analysis
def get_sunglasses_recommendations(face_analysis):
  # Get all sunglasses
  all_sunglasses = get_sunglasses_database()
  
  # Filter models that are recommended for the detected face shape
  recommended_models = [model for model in all_sunglasses if face_analysis["faceShape"] in model["recommendedFor"]]
  
  # If no matches, return a few random models
  if not recommended_models:
      import random
      random.shuffle(all_sunglasses)
      recommended_models = all_sunglasses[:3]
  
  # If we have more than 3 matches, return the top 3
  if len(recommended_models) > 3:
      recommended_models = recommended_models[:3]
  
  return recommended_models

# Function to decode base64 image
def decode_base64_image(base64_string):
  try:
      # Remove data URL prefix if present
      if "base64," in base64_string:
          base64_string = base64_string.split("base64,")[1]
      
      # Decode base64 string to image
      img_data = base64.b64decode(base64_string)
      nparr = np.frombuffer(img_data, np.uint8)
      img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      
      if img is None:
          raise ValueError("Failed to decode image")
          
      return img
  except Exception as e:
      logger.error(f"Error decoding base64 image: {str(e)}")
      raise

# Function to visualize face landmarks for debugging (disabled for production)
def visualize_landmarks(image, landmarks, face_shape, measurements):
    """Simplified version that doesn't save debug images"""
    # Just log the face shape without saving images
    logger.info(f"Face shape determined: {face_shape}")
    # No image saving

# Function to analyze face using MediaPipe
def analyze_face(image):
  # Convert BGR to RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  h, w, _ = image.shape
  
  logger.info(f"Analyzing face in image of size {w}x{h}")
  
  # Process the image
  results = face_mesh.process(image_rgb)
  
  # Check if face is detected
  if not results.multi_face_landmarks:
      logger.warning("No face detected in the image")
      return {"error": "No face detected in the image. Please try again with a clearer photo showing a face."}
  
  # Get the first face
  face_landmarks = results.multi_face_landmarks[0]
  
  # Extract key landmarks for measurements
  # These indices are based on MediaPipe Face Mesh topology
  # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
  
  # Top of forehead (midpoint between eyebrows)
  forehead_top = (face_landmarks.landmark[10].x * w, face_landmarks.landmark[10].y * h)
  
  # Bottom of chin
  chin_bottom = (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h)
  
  # Left and right temples (widest part of forehead)
  temple_left = (face_landmarks.landmark[54].x * w, face_landmarks.landmark[54].y * h)
  temple_right = (face_landmarks.landmark[284].x * w, face_landmarks.landmark[284].y * h)
  
  # Left and right cheekbones (widest part of face)
  cheekbone_left = (face_landmarks.landmark[111].x * w, face_landmarks.landmark[111].y * h)
  cheekbone_right = (face_landmarks.landmark[340].x * w, face_landmarks.landmark[340].y * h)
  
  # Left and right jawline edges
  jaw_left = (face_landmarks.landmark[58].x * w, face_landmarks.landmark[58].y * h)
  jaw_right = (face_landmarks.landmark[288].x * w, face_landmarks.landmark[288].y * h)
  
  # Jawline angle points (for detecting angular vs rounded jaw)
  jaw_angle_left = (face_landmarks.landmark[172].x * w, face_landmarks.landmark[172].y * h)
  jaw_angle_right = (face_landmarks.landmark[397].x * w, face_landmarks.landmark[397].y * h)
  
  # Midpoint of jawline (for measuring jaw curvature)
  jaw_mid_left = (face_landmarks.landmark[140].x * w, face_landmarks.landmark[140].y * h)
  jaw_mid_right = (face_landmarks.landmark[367].x * w, face_landmarks.landmark[367].y * h)
  
  # Left and right eyes (for eye distance)
  eye_left_outer = (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h)
  eye_left_inner = (face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h)
  eye_right_inner = (face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h)
  eye_right_outer = (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h)
  
  # Collect key landmarks for visualization
  key_landmarks = [
      forehead_top, chin_bottom, 
      temple_left, temple_right,
      cheekbone_left, cheekbone_right,
      jaw_left, jaw_right,
      jaw_angle_left, jaw_angle_right,
      jaw_mid_left, jaw_mid_right,
      eye_left_outer, eye_left_inner, 
      eye_right_inner, eye_right_outer
  ]
  
  # Calculate distances
  def distance(p1, p2):
      return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
  
  # Face height (forehead to chin)
  face_height = distance(forehead_top, chin_bottom)
  
  # Face width at different points
  forehead_width = distance(temple_left, temple_right)
  cheekbone_width = distance(cheekbone_left, cheekbone_right)
  jaw_width = distance(jaw_left, jaw_right)
  
  # Eye measurements
  eye_distance = distance(eye_left_inner, eye_right_inner)
  eye_width = distance(eye_left_outer, eye_right_outer)
  
  # Calculate jaw angle (to determine if jaw is angular or rounded)
  # We'll use the angle between three points: jaw_angle_left, chin_bottom, jaw_angle_right
  def calculate_angle(p1, p2, p3):
      # Calculate vectors
      v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
      v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
      
      # Calculate dot product and magnitudes
      dot_product = np.dot(v1, v2)
      magnitude_v1 = np.linalg.norm(v1)
      magnitude_v2 = np.linalg.norm(v2)
      
      # Calculate angle in radians and convert to degrees
      angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
      angle_deg = np.degrees(angle_rad)
      
      return angle_deg
  
  jaw_angle = calculate_angle(jaw_angle_left, chin_bottom, jaw_angle_right)
  
  # Calculate jaw curvature (to help distinguish between round and oval faces)
  # We'll measure how much the midpoints of the jawline deviate from a straight line
  def calculate_curvature(p1, p2, mid1, mid2):
      # Calculate the straight line between p1 and p2
      line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
      line_length = np.linalg.norm(line_vec)
      
      # Calculate perpendicular distance from midpoints to the line
      def perpendicular_distance(point, line_start, line_vec, line_length):
          point_vec = np.array([point[0] - line_start[0], point[1] - line_start[1]])
          # Project point_vec onto line_vec
          projection = np.dot(point_vec, line_vec) / line_length
          # Calculate the projected point on the line
          projected_point = np.array([
              line_start[0] + (projection * line_vec[0] / line_length),
              line_start[1] + (projection * line_vec[1] / line_length)
          ])
          # Calculate distance from point to projected point
          return np.linalg.norm(np.array([point[0] - projected_point[0], point[1] - projected_point[1]]))
      
      # Calculate the average deviation of midpoints from the straight line
      mid1_dist = perpendicular_distance(mid1, p1, line_vec, line_length)
      mid2_dist = perpendicular_distance(mid2, p1, line_vec, line_length)
      
      # Return the average deviation normalized by the line length
      return (mid1_dist + mid2_dist) / (2 * line_length)
  
  jaw_curvature = calculate_curvature(jaw_left, jaw_right, jaw_mid_left, jaw_mid_right)
  
  # Calculate cheek fullness (for round face detection)
  # We'll use the ratio of cheekbone width to jaw width
  cheek_fullness = cheekbone_width / jaw_width
  
  # Calculate key ratios
  face_width = max(forehead_width, cheekbone_width, jaw_width)
  width_to_height_ratio = face_width / face_height
  
  # Normalized width ratios (relative to face width)
  forehead_ratio = forehead_width / face_width
  cheekbone_ratio = cheekbone_width / face_width
  jaw_ratio = jaw_width / face_width
  
  # Calculate chin prominence (for heart shape detection)
  chin_to_cheek_ratio = jaw_width / cheekbone_width
  
  # Calculate face roundness score (new)
  # Higher values indicate a rounder face
  roundness_score = (jaw_curvature * 10) + (cheek_fullness * 0.5) + (abs(1 - width_to_height_ratio) * -2)
  
  # Log all measurements and ratios for debugging
  logger.info("=== FACE MEASUREMENTS ===")
  logger.info(f"Face Height: {face_height:.2f} pixels")
  logger.info(f"Forehead Width: {forehead_width:.2f} pixels")
  logger.info(f"Cheekbone Width: {cheekbone_width:.2f} pixels")
  logger.info(f"Jaw Width: {jaw_width:.2f} pixels")
  logger.info(f"Eye Distance: {eye_distance:.2f} pixels")
  logger.info(f"Eye Width: {eye_width:.2f} pixels")
  logger.info(f"Jaw Angle: {jaw_angle:.2f} degrees")
  logger.info(f"Jaw Curvature: {jaw_curvature:.4f}")
  logger.info(f"Cheek Fullness: {cheek_fullness:.2f}")
  logger.info(f"Chin to Cheek Ratio: {chin_to_cheek_ratio:.2f}")
  logger.info(f"Roundness Score: {roundness_score:.2f}")
  
  logger.info("=== FACE RATIOS ===")
  logger.info(f"Width/Height Ratio: {width_to_height_ratio:.2f}")
  logger.info(f"Forehead Ratio: {forehead_ratio:.2f}")
  logger.info(f"Cheekbone Ratio: {cheekbone_ratio:.2f}")
  logger.info(f"Jaw Ratio: {jaw_ratio:.2f}")
  
  # Improved face shape determination with refined thresholds
  # Default to Oval as it's the most common
  face_shape = "Oval"
  
  # Log the decision process for face shape determination
  logger.info("=== FACE SHAPE DETERMINATION ===")
  
  # ROUND: Soft jaw + similar width and height + high jaw curvature + high cheek fullness
  if (jaw_angle >= 155 and  # Softer, more rounded jaw (larger angle)
      jaw_curvature > 0.04 and  # High jaw curvature
      cheek_fullness > 1.05 and  # Full cheeks
      abs(width_to_height_ratio - 1.0) < 0.2 and  # Width similar to height
      roundness_score > 0.3):  # High roundness score
      face_shape = "Round"
      logger.info("ROUND: Soft jaw with full cheeks and similar width and height")
      logger.info(f"- Jaw angle: {jaw_angle:.2f} >= 155 (softer jaw)")
      logger.info(f"- Jaw curvature: {jaw_curvature:.4f} > 0.04 (curved jawline)")
      logger.info(f"- Cheek fullness: {cheek_fullness:.2f} > 1.05 (full cheeks)")
      logger.info(f"- Width/height ratio: {width_to_height_ratio:.2f} close to 1.0")
      logger.info(f"- Roundness score: {roundness_score:.2f} > 0.3")
  
  # SQUARE: Angular jaw + similar width at forehead, cheekbones, and jaw
  elif (jaw_angle < 160 and  # More angular jaw (adjusted threshold)
        abs(forehead_ratio - jaw_ratio) < 0.2 and  # More tolerance
        abs(cheekbone_ratio - jaw_ratio) < 0.15 and  # Similar widths at cheekbones and jaw
        abs(width_to_height_ratio - 1.0) < 0.25):  # More tolerance for width/height
      face_shape = "Square"
      logger.info("SQUARE: Angular jaw with similar widths at forehead, cheekbones, and jaw")
      logger.info(f"- Jaw angle: {jaw_angle:.2f} < 160 (angular jaw)")
      logger.info(f"- Forehead to jaw difference: {abs(forehead_ratio - jaw_ratio):.2f} < 0.2")
      logger.info(f"- Cheekbone to jaw difference: {abs(cheekbone_ratio - jaw_ratio):.2f} < 0.15")
      logger.info(f"- Width/height ratio: {abs(width_to_height_ratio - 1.0):.2f} < 0.25")
  
  # OBLONG/RECTANGLE: Longer face with angular jaw and similar widths
  elif (width_to_height_ratio < 0.85 and  # Longer face (even stricter threshold)
        jaw_angle < 160 and  # Angular jaw
        abs(forehead_ratio - jaw_ratio) < 0.25 and abs(cheekbone_ratio - jaw_ratio) < 0.15):  # More tolerance for width differences
      face_shape = "Oblong"
      logger.info("OBLONG: Longer face with angular jaw and similar widths")
      logger.info(f"- Width/height ratio: {width_to_height_ratio:.2f} < 0.85 (longer face)")
      logger.info(f"- Jaw angle: {jaw_angle:.2f} < 160 (angular jaw)")
      logger.info(f"- Forehead to jaw difference: {abs(forehead_ratio - jaw_ratio):.2f} < 0.25")
      logger.info(f"- Cheekbone to jaw difference: {abs(cheekbone_ratio - jaw_ratio):.2f} < 0.15")
  
  # HEART: Wider at forehead, narrower at jaw with pointed chin
  elif (forehead_ratio > cheekbone_ratio > jaw_ratio and  # Tapering from forehead to jaw
        forehead_ratio - jaw_ratio > 0.15 and  # Significant difference between forehead and jaw
        chin_to_cheek_ratio < 0.8):  # Narrow chin compared to cheekbones
      face_shape = "Heart"
      logger.info("HEART: Wider at forehead, narrower at jaw with pointed chin")
      logger.info(f"- Forehead > cheekbone > jaw: {forehead_ratio:.2f} > {cheekbone_ratio:.2f} > {jaw_ratio:.2f}")
      logger.info(f"- Forehead to jaw difference: {forehead_ratio - jaw_ratio:.2f} > 0.15")
      logger.info(f"-  Forehead to jaw difference: {forehead_ratio - jaw_ratio:.2f} > 0.15")
      logger.info(f"- Chin to cheek ratio: {chin_to_cheek_ratio:.2f} < 0.8 (narrow chin)")
  
  # TRIANGLE: Narrower at forehead, wider at jaw
  elif (jaw_ratio > cheekbone_ratio > forehead_ratio and  # Widening from forehead to jaw
        jaw_ratio - forehead_ratio > 0.15):  # Significant difference between jaw and forehead
      face_shape = "Triangle"
      logger.info("TRIANGLE: Narrower at forehead, wider at jaw")
      logger.info(f"- Jaw > cheekbone > forehead: {jaw_ratio:.2f} > {cheekbone_ratio:.2f} > {forehead_ratio:.2f}")
      logger.info(f"- Jaw to forehead difference: {jaw_ratio - forehead_ratio:.2f} > 0.15")
  
  # DIAMOND: Narrow forehead and jaw, wide cheekbones
  elif (cheekbone_ratio > forehead_ratio and 
        cheekbone_ratio > jaw_ratio and 
        abs(forehead_ratio - jaw_ratio) < 0.1):
      face_shape = "Diamond"
      logger.info("DIAMOND: Narrow forehead and jaw, wide cheekbones")
      logger.info(f"- Cheekbone > forehead: {cheekbone_ratio:.2f} > {forehead_ratio:.2f}")
      logger.info(f"- Cheekbone > jaw: {cheekbone_ratio:.2f} > {jaw_ratio:.2f}")
      logger.info(f"- Forehead to jaw difference: {abs(forehead_ratio - jaw_ratio):.2f} < 0.1")
  
  # OVAL: Balanced proportions with soft jaw and slightly longer face
  # Modified to better distinguish from oblong and round
  elif (width_to_height_ratio >= 0.75 and  # Not too long
        width_to_height_ratio <= 0.9 and  # Not too wide
        jaw_angle >= 150 and  # Softer jaw
        cheekbone_ratio >= jaw_ratio and  # Cheekbones wider than or equal to jaw
        cheekbone_ratio >= forehead_ratio and  # Cheekbones wider than or equal to forehead
        roundness_score <= 0.3):  # Not too round
      face_shape = "Oval"
      logger.info("OVAL: Balanced proportions with soft jaw and slightly longer face")
      logger.info(f"- Width/height ratio: {width_to_height_ratio:.2f} between 0.75 and 0.9")
      logger.info(f"- Jaw angle: {jaw_angle:.2f} >= 150 (softer jaw)")
      logger.info(f"- Cheekbone >= jaw: {cheekbone_ratio:.2f} >= {jaw_ratio:.2f}")
      logger.info(f"- Cheekbone >= forehead: {cheekbone_ratio:.2f} >= {forehead_ratio:.2f}")
      logger.info(f"- Roundness score: {roundness_score:.2f} <= 0.3 (not too round)")
  
  # If none of the above conditions are met, it defaults to Oval
  else:
      logger.info("OVAL: Default face shape (no other conditions met)")
      logger.info(f"- Width/height ratio: {width_to_height_ratio:.2f}")
      logger.info(f"- Forehead ratio: {forehead_ratio:.2f}")
      logger.info(f"- Cheekbone ratio: {cheekbone_ratio:.2f}")
      logger.info(f"- Jaw ratio: {jaw_ratio:.2f}")
      logger.info(f"- Jaw angle: {jaw_angle:.2f} degrees")
      logger.info(f"- Jaw curvature: {jaw_curvature:.4f}")
      logger.info(f"- Roundness score: {roundness_score:.2f}")
  
  logger.info(f"Final determined face shape: {face_shape}")
  
  # Create measurements dictionary for visualization
  measurements_dict = {
      "Width/Height Ratio": width_to_height_ratio,
      "Forehead Ratio": forehead_ratio,
      "Cheekbone Ratio": cheekbone_ratio,
      "Jaw Ratio": jaw_ratio,
      "Jaw Angle": jaw_angle,
      "Jaw Curvature": jaw_curvature,
      "Cheek Fullness": cheek_fullness,
      "Roundness Score": roundness_score
  }
  
  # Visualize landmarks for debugging
  visualize_landmarks(image, key_landmarks, face_shape, measurements_dict)
  
  # Determine eye distance category
  eye_to_face_ratio = eye_distance / face_width
  if eye_to_face_ratio > 0.3:
      eye_distance_category = "Wide-set"
  elif eye_to_face_ratio < 0.2:
      eye_distance_category = "Close-set"
  else:
      eye_distance_category = "Average"
  
  # Determine face width category
  if width_to_height_ratio > 0.95:
      face_width_category = "Wide"
  elif width_to_height_ratio < 0.85:
      face_width_category = "Narrow"
  else:
      face_width_category = "Average"
  
  # Estimate skin tone
  # Sample a few points on the face for skin tone
  skin_points = [
      (int(forehead_top[0]), int(forehead_top[1]) + 20),  # Slightly below forehead
      (int(cheekbone_left[0] + 20), int(cheekbone_left[1])),  # Left cheek
      (int(cheekbone_right[0] - 20), int(cheekbone_right[1]))  # Right cheek
  ]
  
  # Calculate average color
  skin_samples = []
  for point in skin_points:
      if 0 <= point[0] < w and 0 <= point[1] < h:  # Check if point is within image bounds
          skin_samples.append(image[point[1], point[0]])
  
  if skin_samples:
      avg_color = np.mean(skin_samples, axis=0)
      brightness = np.mean(avg_color)
      
      if brightness > 200:
          skin_tone = "Fair"
      elif brightness > 150:
          skin_tone = "Medium"
      elif brightness > 100:
          skin_tone = "Olive"
      else:
          skin_tone = "Deep"
  else:
      skin_tone = "Medium"  # Default
  
  # Get explanation for face shape
  explanation = get_explanation_for_face_shape(face_shape)
  
  # Create face analysis result with detailed measurements
  face_analysis_result = {
      "faceShape": face_shape,
      "skinTone": skin_tone,
      "faceWidth": face_width_category,
      "eyeDistance": eye_distance_category,
      "dominantEmotion": "neutral",  # MediaPipe doesn't provide emotion detection
      "explanation": explanation,
      "gender": "Unknown",  # MediaPipe doesn't provide gender detection
      "age": 25,  # MediaPipe doesn't provide age detection
      "measurements": {
          "faceHeight": float(face_height),
          "faceWidth": float(face_width),
          "foreheadWidth": float(forehead_width),
          "cheekboneWidth": float(cheekbone_width),
          "jawWidth": float(jaw_width),
          "eyeDistance": float(eye_distance),
          "widthToHeightRatio": float(width_to_height_ratio),
          "foreheadRatio": float(forehead_ratio),
          "cheekboneRatio": float(cheekbone_ratio),
          "jawRatio": float(jaw_ratio),
          "jawAngle": float(jaw_angle),
          "jawCurvature": float(jaw_curvature),
          "cheekFullness": float(cheek_fullness),
          "roundnessScore": float(roundness_score),
          "chinToCheekRatio": float(chin_to_cheek_ratio)
      }
  }
  
  return face_analysis_result

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face_endpoint():
  try:
      logger.info("Received analyze-face request")
      
      # Get image data from request
      if 'image' not in request.files and 'imageData' not in request.json:
          logger.warning("No image provided in request")
          return jsonify({"error": "No image provided", "message": "Please upload a photo or take a picture with your camera."}), 400
      
      # Process image from file or base64 data
      if 'image' in request.files:
          logger.info("Processing image from file upload")
          file = request.files['image']
          img_data = file.read()
          nparr = np.frombuffer(img_data, np.uint8)
          image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      else:
          logger.info("Processing image from base64 data")
          base64_image = request.json['imageData']
          image = decode_base64_image(base64_image)
      
      if image is None:
          logger.error("Failed to decode image")
          return jsonify({"error": "Failed to decode image", "message": "The image could not be processed. Please try a different image."}), 400
          
      logger.info(f"Image decoded successfully, shape: {image.shape}")
      
      # Analyze face
      analysis_result = analyze_face(image)
      
      # Check for error
      if 'error' in analysis_result:
          logger.warning(f"Face analysis error: {analysis_result['error']}")
          return jsonify(analysis_result), 400
      
      logger.info(f"Face analysis successful, detected shape: {analysis_result['faceShape']}")
      
      # Get sunglasses recommendations
      recommendations = get_sunglasses_recommendations(analysis_result)
      logger.info(f"Generated {len(recommendations)} recommendations")
      
      # Prepare response
      response = {
          "userImage": request.json.get('imageData') if 'imageData' in request.json else None,
          **analysis_result,
          "recommendations": recommendations,
          "success": True  # Add success flag for frontend to trigger confetti
      }
      
      return jsonify(response)
  
  except Exception as e:
      error_details = traceback.format_exc()
      logger.error(f"Error in analyze-face endpoint: {str(e)}\n{error_details}")
      return jsonify({
          "error": "Failed to analyze face", 
          "message": "An unexpected error occurred while analyzing your face. Please try again with a different photo.",
          "details": str(e)
      }), 500

@app.route('/api/more-recommendations', methods=['POST'])
def more_recommendations_endpoint():
  try:
      logger.info("Received more-recommendations request")
      
      # Get face shape and current recommendations from request
      data = request.json
      face_shape = data.get('faceShape')
      current_ids = data.get('currentIds', [])
      
      if not face_shape:
          logger.warning("No face shape provided in request")
          return jsonify({"error": "Face shape is required", "message": "Face shape information is missing. Please analyze your face first."}), 400
      
      logger.info(f"Getting more recommendations for face shape: {face_shape}, excluding IDs: {current_ids}")
      
      # Get all sunglasses
      all_sunglasses = get_sunglasses_database()
      
      # Filter models that are recommended for the face shape and not already shown
      recommended_models = [
          model for model in all_sunglasses 
          if face_shape in model["recommendedFor"] and model["id"] not in current_ids
      ]
      
      # If no matches, return random models not already shown
      if not recommended_models:
          logger.info("No matching recommendations found, returning random models")
          import random
          recommended_models = [model for model in all_sunglasses if model["id"] not in current_ids]
          random.shuffle(recommended_models)
      
      # Return up to 3 more recommendations
      more_recommendations = recommended_models[:3]
      logger.info(f"Returning {len(more_recommendations)} more recommendations")
      
      return jsonify({"recommendations": more_recommendations, "success": True})
  
  except Exception as e:
      error_details = traceback.format_exc()
      logger.error(f"Error in more-recommendations endpoint: {str(e)}\n{error_details}")
      return jsonify({
          "error": "Failed to get more recommendations", 
          "message": "An unexpected error occurred while getting more recommendations. Please try again.",
          "details": str(e)
      }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
  logger.info("Received health check request")
  return jsonify({"status": "ok", "message": "Server is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
