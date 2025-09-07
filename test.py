import pygame
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from constants import PYGAME_UI as PYGCFG
from constants import MODEL

pygame.init()
fps = 60
timer = pygame.time.Clock()

# Menu & UI
font = pygame.font.SysFont(None, 36)
canvas = pygame.Surface((PYGCFG.WIDTH, PYGCFG.HEIGHT))
canvas.fill("white")
screen = pygame.display.set_mode([PYGCFG.WIDTH, PYGCFG.HEIGHT])
pygame.display.set_caption("Wywy Recognition")

# runtime/data variables
prediction = None
running = True

models = []
if MODEL.GET_ALL_MODELS:
    for model_name in os.listdir(MODEL.MODELS_PATH):
        if model_name.endswith(".keras"):
            models.append(keras.models.load_model(os.path.join(MODEL.MODELS_PATH, model_name)))
else:
    for model_name in MODEL.MODELS_NAMES:
        models.append(keras.models.load_model(os.path.join(MODEL.MODELS_PATH, model_name) + ".keras"))
        
default_model = keras.models.load_model(MODEL.DEFAULT_MODEL_PATH)

def draw_menu():
    pygame.draw.rect(screen, "gray", [0, 0, PYGCFG.WIDTH, 70])
    pygame.draw.line(screen, 'black', (0, 70), (PYGCFG.WIDTH, 70), 3)
    
    if prediction is not None:
        pred_text = font.render(f"Prediction: {prediction}", True, (255, 255, 0))
        screen.blit(pred_text, (10, 40)) # @todo

def get_prediction():
    global prediction
    
    # Convert canvas to numpy image
    data = pygame.image.tostring(canvas, "RGB")     # canvas is RGB
    img = Image.frombytes("RGB", (PYGCFG.WIDTH, PYGCFG.HEIGHT), data)
    # img = img.convert("L")
    img = img.resize(MODEL.IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, *MODEL.IMAGE_SIZE, 3)
    
    # eat the image up and generate a prediction
    prediction = default_model.predict(arr, verbose=0)
    
    # (verbose) print out 
    if PYGCFG.VERBOSE:
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
        pygame.draw.circle(screen, PYGCFG.BRUSH_PREVIEW_COLOR, mouse, PYGCFG.BRUSH_SIZE)
        
        # if the user is trying to draw,
        if left_click:
            if prev_pos is not None:
                pygame.draw.line(canvas, "black", prev_pos, mouse, PYGCFG.BRUSH_SIZE * 2)
            else:
                pygame.draw.circle(canvas, "black", (mouse[0], mouse[1]), PYGCFG.BRUSH_SIZE)
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