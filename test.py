import pygame
import os
import numpy as np
from PIL import Image
from tensorflow import keras

pygame.init()
fps = 60
timer = pygame.time.Clock()

VERBOSE = True

# Menu & UI
WIDTH = 255
HEIGHT = 255 + 70 + 3
font = pygame.font.SysFont(None, 36)
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill("white")

# brush settings
BRUSH_SIZE = 5
BRUSH_COLOR = "black"
BRUSH_PREVIEW_COLOR = (136, 138, 137)

screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Wywy Recognition")

# ML constants
IMAGE_SHAPE = (128, 128, 1)
IMAGE_SIZE = (128, 128)
MODELS_PATH = "models"
DEFAULT_MODEL_PATH = "models/epoch_13.keras"
GET_ALL_MODELS = False
MODELS_NAMES = ["epoch_13", "epoch_14"]

# runtime/data variables
prediction = None
running = True

models = []
if GET_ALL_MODELS:
    for model_name in os.listdir(MODELS_PATH):
        if model_name.endswith(".keras"):
            models.append(keras.models.load_model(os.path.join(MODELS_PATH, model_name)))
else:
    for model_name in MODELS_NAMES:
        models.append(keras.models.load_model(os.path.join(MODELS_PATH, model_name) + ".keras"))
        
default_model = keras.models.load_model(DEFAULT_MODEL_PATH)

def draw_menu():
    pygame.draw.rect(screen, "gray", [0, 0, WIDTH, 70])
    pygame.draw.line(screen, 'black', (0, 70), (WIDTH, 70), 3)
    
    if prediction is not None:
        pred_text = font.render(f"Prediction: {prediction}", True, (255, 255, 0))
        screen.blit(pred_text, (10, 40)) # @todo

def get_prediction():
    global prediction
    
    # Convert canvas to numpy image
    data = pygame.image.tostring(canvas, "RGB")     # canvas is RGB
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), data)
    # img = img.convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, *IMAGE_SIZE, 3)
    
    # eat the image up and generate a prediction
    prediction = default_model.predict(arr, verbose=0)
    
    # (verbose) print out 
    if VERBOSE:
        print("PREDICTIONS:\n")
        for model in models:
            current_prediction = model.predict(arr, verbose=0)
            print(" * ", round(current_prediction[0][0] * 100, 0), "% Other; ", round(current_prediction[0][1] * 100, 0), "% Wywy.", sep="")
        
        print("\n\n")

# main loop
while running:
    timer.tick(fps)
    screen.fill("white")
    # get mouse input information
    mouse = pygame.mouse.get_pos()
    left_click = pygame.mouse.get_pressed()[0]
    
    # if the mouse is outside the information bar,
    if mouse[1] > 70:
        # create a preview of what the brush will paint
        pygame.draw.circle(screen, BRUSH_PREVIEW_COLOR, mouse, BRUSH_SIZE)
        
        # if the user is trying to draw,
        if left_click:
            if prev_pos is not None:
                pygame.draw.line(canvas, "black", prev_pos, mouse, BRUSH_SIZE * 2)
            else:
                pygame.draw.circle(canvas, "black", (mouse[0], mouse[1]), BRUSH_SIZE)
            prev_pos = pygame.mouse.get_pos()
        else:
            prev_pos = None
    
    draw_menu()
    screen.blit(canvas, (0, 0))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                get_prediction()
            if event.key == pygame.K_c:
                canvas.fill("white")
                prediction = None
    
    pygame.display.flip()

pygame.quit()