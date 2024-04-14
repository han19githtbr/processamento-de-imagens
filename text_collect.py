import cv2
import mahotas
import matplotlib.pyplot as plt
import math
import pytesseract

def print_image_wait(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)

def BinarizationImg(img):
    T = mahotas.thresholding.otsu(img)
    bin = img.copy()
    bin[bin > T] = 255
    bin[bin < 255] = 0
    return cv2.bitwise_not(bin)

def draw_contours(img, objects):
    objects_length = len(objects)
    for c in range(objects_length):
        cv2.drawContours(img, objects,c,(0,128,0),2)


# Abrir imagem alvo
target_img = cv2.imread('asserts/alpha.png')

#open alphabet image
alpha_img = cv2.imread('asserts/alpha.png')

# Redimensionar imagem alvo
target_img = cv2.resize(target_img, (0,0), fx = 2.0, fy = 2.0)

# Converter imagem para escala de cinza
target_img_grayscale = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
alpha_img_grayscale = cv2.cvtColor(alpha_img, cv2.COLOR_BGR2GRAY)

# Exibir imagem alvo em escala de cinza
plt.imshow(target_img_grayscale, cmap='gray')
plt.title('GrayScaler')
plt.show()

# Aplicar filtro bilateral
target_smooths = cv2.bilateralFilter(target_img_grayscale, 10, 75, 75)
alpha_smooths = cv2.bilateralFilter(alpha_img_grayscale, 10, 75, 75)

# Exibir imagem suavizada com filtro bilateral
plt.imshow(target_smooths)
plt.title('Bilateral filter')
plt.show()

# Binarização resultando em pixels brancos e pretos
target_bin = BinarizationImg(target_smooths)
alpha_bin = BinarizationImg(alpha_smooths)

plt.imshow(target_bin)
plt.title('Binarization')
plt.show()

# Detectar bordas usando Canny
target_edges = cv2.Canny(target_bin, 70, 150)
alpha_edges = cv2.Canny(alpha_bin, 70, 150)

plt.imshow(target_edges)
plt.title('Canny')
plt.show()

# Identificar contornos
target_contours, target_hierarchy = cv2.findContours(target_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
alpha_contours, alpha_hierarchy = cv2.findContours(alpha_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


plt.imshow(target_img)
plt.title('Target')


# Loop sobre os contornos
for c in target_contours:
    x, y, w, h = cv2.boundingRect(c)
    curt = target_img[y:y+h, x:x+w]
    # Exibir a letra recortada dentro do loop
    plt.imshow(curt, cmap='gray')
    plt.title('Curt')
    plt.show(block=False)  # Defina block=False para que a execução do código continue imediatamente
    plt.pause(0.01)  # Aguarde um curto período de tempo para verificar se a janela foi fechada
    # Aguarde até que a janela seja fechada pelo usuário
    while plt.fignum_exists(1):
        plt.pause(0.1)  # Aguarde 100ms antes de verificar novamente
    plt.close()  # Feche a janela atual antes de continuar para a próxima letra

