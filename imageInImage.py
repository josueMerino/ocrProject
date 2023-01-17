import cv2

# Cargar la imagen donde se buscará la otra imagen
big_image = cv2.imread('images/ground.jpg')

# Cargar la imagen a buscar dentro de la otra imagen
small_image = cv2.imread('images/monaLisa.jpg')

# Buscar la imagen dentro de la otra imagen utilizando el método matchTemplate
result = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)

# Obtener las coordenadas (x, y) del área donde se encuentra la imagen
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Dibujar un rectángulo alrededor de la imagen encontrada
top_left = max_loc
bottom_right = (top_left[0] + small_image.shape[1],
                top_left[1] + small_image.shape[0])
cv2.rectangle(big_image, top_left, bottom_right, (255, 0, 0), 5)

# Mostrar la imagen con el rectángulo dibujado
cv2.imshow('Image with rectangle', big_image)

# Esperar a que el usuario presione una tecla para continuar
cv2.waitKey(0)

# Exportar la imagen con el rectángulo dibujado
cv2.imwrite('out/result.jpg', big_image)

# Cerrar la ventana
cv2.destroyAllWindows()
