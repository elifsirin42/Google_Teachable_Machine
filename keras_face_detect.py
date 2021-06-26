import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Netlik için bilimsel gösterimi devre dışı bırakılıyor.
np.set_printoptions(suppress=True)

# Geliştirilen model yükleniyor.
model = tensorflow.keras.models.load_model('keras_model.h5')

# Keras modeline beslemek için doğru şeklin dizisini oluşturuluyor.
# Diziye koyabileceğiniz 'uzunluk' veya görüntü sayısı belirleniyor.

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Test edilecek image görüntüsü açılır.
image = Image.open('test_photo_maskesiz2.jpg')

#TM2'dekiyle aynı stratejiyle görüntü 224x224 boyutunda yeniden boyutlandırılır.
#resmi en az 224x224 olacak şekilde yeniden boyutlandırma ve ardından merkezden kırpma gerçekleştirilir.
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#Görüntü bir numpy dizisine dönüştürülür.
image_array = np.asarray(image)

# Yeniden boyutlandırılmış resim gösterilir.
image.show()

# Resim normalize edilir.
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Normalize edilmiş görüntü diziye yüklenir.
data[0] = normalized_image_array

# Modele göre tahminleme işlemi gerçekleştirilir.
prediction = model.predict(data)

#Tahmin sonucu konsola yazdırılır.
print(prediction)

#Tahmin sonucuna yüklenen görüntünün maskeli veya maskesiz olduğu çıkarımı konsola yazdırılır.
if(prediction[0][0] > prediction[0][1] ):
    print("Maskeli Birey")
else:
    print("Maskesiz Birey")


