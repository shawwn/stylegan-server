(import sys)
(import os)
(import json)
(import redis)
(from urllib.parse import urlparse)

(defvar items* (obj))
(defconst images* (obj))

(def make-redis (url)
  (let (url (or url (get os.environ "REDIS_URL"))
        o (urlparse url))
    (redis.Redis host: o.hostname port: o.port password: o.password)))

(def on-err (handler thunk)
  (let ((ok v) (guard (thunk)))
    (if ok v (handler v))))

(mac errsafe (expr)
  `(on-err (fn (c) nil)
           (fn () ,expr)))

(def start ()
  (defvar r (make-redis))
  r)

(def max-id ()
  (r.incrby "max_id" 0))
  
(def load-item (id)
  (if (has? items* id)
      (get items* id)
    (let-when s (r.get (cat "item?id=" (tostr id)))
      (with i (json.loads s)
        (set (get items* id) i)))))

(def load-items ()
  (with n (max-id)
    (for i in (range 1 n)
      (load-item i))))

(def item (id)
  (or (has items* id)
      (errsafe (load-item id))))

(mac ivar (i k) `(get (item ,i) ',k))

(import reader)

(def read-safe (s)
  (if (string? s)
      (either (reader.maybe-number s) s)
    s))

(def atom (x)
  (if (array? x) (none? x) (is? x)))

(def alist (x)
  (not (atom x)))

(def acons (x)
  (or (array? x) (obj? x)))

(def no (x)
  (if (array? x) (none? x) (nil? x)))

(def alist (x)
  (or (no x) (acons x)))

(def car (x)
  (if (alist x) (hd x) x))

(def cdr (x)
  (if (alist x)
      (let l (tl x)
        (if (and (none? l)
                 (has? l 'rest))
            (get l 'rest)
          l))
    x))

(def cons (x y)
  (if (nil? y)
      `(,x)
      (acons y)
      `(,x ,@y)
    `(,x rest: ,y)))

(def treewise (f base tree)
  (if (atom tree)
      (base tree)
      (f (treewise f base (car tree)) 
         (treewise f base (cdr tree)))))

(def maptree (f base l)
  (if (atom l)
      (base l)
    (with r (dupe l)
      (each (k v) (f l)
        (set (get r k) (maptree f base v))))))

(def idfn (x) x)

(import requests)

(def GET (url)
  (requests.get url))

(def hd? (l x)
  (= (errsafe (hd l)) x))

(import PIL.Image)
(from io import BytesIO)

(def mkdirp args
  (with p (apply os.path.join args)
    (os.makedirs p exist_ok: true)))

(def image-cache (url img)
  (let (img2 (img.convert "RGB")
        id (image-id url)
        dname (mkdirp "/tmp" "cache" id)
        fname (os.path.join (mkdirp dname "raw") "image.jpg")
        aname (os.path.join (mkdirp dname "aligned") "image.png")
        aligned nil)
    (img2.save fname "JPEG" quality: 95)
    (set aligned (if (os.path.exists aname)
                     (PIL.Image.open aname)
                     (face-align fname)))
    (if aligned
        (aligned.save aname "PNG")
      (set aname nil))
    (list 'face id: id raw-image: fname aligned-image: aname path: dname)))

(def load-image (url)
  (or (has images* url)
      (let resp (GET url)
        (when (= resp.status_code 200)
          (let (img (PIL.Image.open (BytesIO resp.content))
                h (image-cache url img))
            (set (get images* url) h)
            (set (get images* (shash url)) h)
            h)))))

(def image (url)
  (has (load-image url) 'id))

(mac imgvar (id k)
  `(has (load-image ,id) ',k))

(import hashlib)

(def shash (s)
  (let (s (if (string? s)
              (s.encode "utf-8")
             s)
        h (hashlib.md5 s))
    (h.hexdigest)))

(def image-id (url)
  (shash url))

(def image-filename (url)
  (get (load-image url) 'filename))

(def read-image (l)
  (if (hd? l 'image)
      (let url (at l 1)
        (let-when img (load-image url)
          (list 'ref (get img 'id))))
    l))

(def read-op (i)
  (maptree read-image read-safe (ivar i op)))

(def save-item (tem)
  (let id (or (has tem 'id) (r.incrby "maxId" 1))
    (set (get tem 'id) id)
    (r.set (cat "item?id=" (tostr id)) (json.dumps tem))
    id))

(import os)
(import sys)
(import bz2)
(import argparse)
(from keras.utils import get_file)
(from ffhq_dataset.face_alignment import image_align)
(from ffhq_dataset.landmarks_detector import LandmarksDetector)
(import multiprocessing)

(defconst LANDMARKS_MODEL_URL "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

(def unpack-bz2 (src-path)
  (let (f (bz2.BZ2File src_path)
        data (f.read)
        dst-path (get src-path (: -4)))
    (with (open dst-path "wb") as fp
      (fp.write data))
    dst-path))

(defvar landmarks-model-path*
  (unpack_bz2 (get_file "shape_predictor_68_face_landmarks.dat.bz2" LANDMARKS_MODEL_URL cache_subdir: "temp")))

(defvar landmarks-detector* (LandmarksDetector landmarks-model-path*))

(def get-landmarks (raw-img-path)
  (next (landmarks-detector*.get-landmarks raw-img-path)))

(def face-align (raw-img-path)
  (let marks (get-landmarks raw-img-path)
    (if (none? marks)
        (list 'error "No face detected")
      (image-align raw-img-path marks))))

