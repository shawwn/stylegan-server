from PIL import Image, ExifTags

def square_thumb(img, thumb_size):
  THUMB_SIZE = (thumb_size,thumb_size)
  if img.getexif():
    exif=dict((ExifTags.TAGS[k], v) for k, v in img.getexif().items() if k in ExifTags.TAGS)
    if exif['Orientation'] == 3 :
      img=img.rotate(180, expand=True)
    elif exif['Orientation'] == 6 :
      img=img.rotate(270, expand=True)
    elif exif['Orientation'] == 8 :
      img=img.rotate(90, expand=True)


  width, height = img.size

  # square it

  if width > height:
    delta = width - height
    left = int(delta/2)
    upper = 0
    right = height + left
    lower = height
  else:
    delta = height - width
    left = 0
    upper = int(delta/2)
    right = width
    lower = width + upper

  img = img.crop((left, upper, right, lower))
  img.thumbnail(THUMB_SIZE, Image.ANTIALIAS)

  return img.resize(THUMB_SIZE, resample=Image.LANCZOS)


