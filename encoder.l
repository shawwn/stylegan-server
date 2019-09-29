(import sys)
(import os)
(import json)
(import redis)
(from urllib.parse import urlparse)

(def idfn (x) x)

(def hd? (l x)
  (= (errsafe (hd l)) x))

(def maptree (f base l)
  (if (atom? l)
      (base l)
    (with r (dupe l)
      (each (k v) (f l)
        (set (get r k) (maptree f base v))))))

(def p parts
  (apply os.path.join parts))

(def to-bytes (x)
  (if (string? x) (x.encode "utf-8") x))

(define-special b (x)
  (cat "b" (compile x)))

(def on-err (handler thunk)
  (let ((ok v) (guard (thunk)))
    (if ok v (handler v))))

(mac errsafe (expr)
  `(on-err (fn (c) nil)
           (fn () ,expr)))

(def make-redis (url)
  (let (url (or url (get os.environ "REDIS_URL"))
        o (urlparse url))
    (redis.Redis host: o.hostname port: o.port password: o.password)))

(defvar r* (make-redis))

(def load-item (id)
  (let-when x (r*.get (cat id "_faces"))
    (with r (json.loads x)
      (set (get r 'id) id))))

(defconst items* (obj))

(def item (id rest: ks)
  (if (obj? id) (set id (has id 'id)))
  (with i (or (has items* id)
              (load-item id))
    (when i (set (get items* id) i))
    (while (and i (some? ks))
      (set i (has i (hd ks)))
      (set ks (tl ks)))))

(def maxid ()
  (r*.incrby "max_id" 0))

(mac for-each-item (i rest: body)
  (let-unique (n)
    `(let ,n (maxid)
       (for id ,n
         (let-when ,i (item id)
           ,@body)))))

(import hashlib)

(def shash (s)
  (let (s (if (string? s)
              (s.encode "utf-8")
             s)
        h (hashlib.md5 s))
    (h.hexdigest)))
  
(defvar encoder-cache-dir* "/tmp/cache")
(defvar encoder-open-url-verbose* false)

(import dnnlib)

(def open-url (url)
  (dnnlib.util.open_url url verbose: encoder-open-url-verbose* cache_dir: encoder-cache-dir*))

(import requests)

(def urlbytes (url)
  (let-when r (requests.get url)
    (to-bytes r.content)))

;(def urlbytes (url)
; (with (open-url url) as f
;   (to-bytes (f.read))))

(def hash-url (url)
  (shash (urlbytes url)))

(def todisk (fname value)
  (if (nil? value)
      (errsafe (os.remove fname))
    (do (with (open (cat fname ".tmp") "wb") as f
          (f.write (to-bytes value)))
        (os.rename (cat fname ".tmp") fname))))

(def url-key (url k)
  (cat (hash-url url) "_" (compile k)))

(def url-key-filename (url k)
  (p encoder-cache-dir* (url-key url k)))

(def url-key-exists (url k)
  (os.path.exists (url-key-filename url k)))

(def put-url-key (url k v)
  (let fname (url-key-filename url k)
    (todisk fname v)))

(def get-url-key (url k)
  (let fname (url-key-filename url k)
    (errsafe (with (open fname "rb") as f (f.read)))))

(import tempfile)

(mac with-temp-dir (var rest: body)
  `(with (tempfile.TemporaryDirectory suffix: ',(compile (cat "-" var))) as ,var
     ,@body))

(import PIL.Image)
(from io import BytesIO)

(def image-from-bytes (s)
  (PIL.Image.open (BytesIO (to-bytes s))))

(def image-to-bytes (img rest: args)
  (with-temp-dir tmp-image-dir
    (let fname (p tmp-image-dir "image.jpg")
      (img.save fname args)
      (with (open fname "rb") as f
        (f.read)))))

(def url-image (url)
  (PIL.Image.open (BytesIO (urlbytes url))))

(import scrap)

(from glob import glob)

(def align-image (url)
  (with-temp-dir aligned-src-dir
    (with-temp-dir aligned-dst-dir
      (let img (url-image url)
        (set img (img.convert "RGB"))
        (img.save (p aligned-src-dir "image.jpg") "JPEG")
        (scrap.system "python3" (list "align_images.py" aligned-src-dir aligned-dst-dir))
        (let-when aligned-fname (hd (glob (p aligned-dst-dir "*")))
          (PIL.Image.open aligned-fname))))))

(def aligned-image (url)
  (let-when bs (get-url-key url 'aligned)
    (return (image-from-bytes bs)))
  (let-when img (align-image url)
    (let-when bs (image-to-bytes img)
      (put-url-key url 'aligned bs))
    img))

(def json-from-bytes (s)
  (json.loads (to-bytes s)))

(def json-to-bytes (h)
  (json.dumps h))

(def obj-from-bytes (s)
  (json.loads (to-bytes s)))

(def obj-to-bytes (h)
  (if (isinstance h np.ndarray)
    (set h (h.tolist)))
  (json.dumps h))

(import numpy as np)

(def make-latent-image (img)
  (with-temp-dir latent-src-dir
    (with-temp-dir latent-dst-dir
      (with-temp-dir latent-generated-dir
        (let img (img.copy)
          (set img (img.convert "RGB"))
          (img.save (p latent-src-dir "image.jpg") "JPEG")
          (scrap.system "python3" (list "encode_images.py" latent-src-dir latent-generated-dir latent-dst-dir))
          (let-when latent-fname (hd (glob (p latent-dst-dir "*")))
            (np.load latent-fname)))))))

(def latent-image (url)
  (let-when bs (get-url-key url 'latent)
    (return (obj-from-bytes bs)))
  (let-when latent (make-latent-image url)
    (let-when bs (obj-to-bytes latent)
      (put-url-key url 'latent bs))
    latent))

(def read-image (l)
  (if (hd? l 'image)
      (let url (at l 1)
        (let-when latent (latent-image url)
          (list 'latent latent)))
    l))

(def read-op (i)
  (maptree read-image idfn (item i op)))

(def gather-urls (i)
  (let-when op (item i 'op)
    (with r ()
      (maptree (fn (l)
                 (when (hd? l 'image)
                   (let url (at l 1)
                     (add r url)))
                 l)
               idfn
               op))))

(def all-urls ()
  (with r ()
    (for-each-item i
      (join! r (gather-urls i)))))

(def all-items ()
  (with r ()
    (for-each-item i (add r i))))

(defvar ops* (obj))
(defvar results* (obj))

(def process-item (i)
  (let k (cat (item i 'id) "_faces:result")
    (unless (r*.keys k)
      (print (cat "Processing " k))
      (let-when op (read-op i)
        (set (get ops* id) op)
        (set (get results* id) (eval op))))))

(def process-items ()
  (for-each-item i
    (errsafe (process-item i))))