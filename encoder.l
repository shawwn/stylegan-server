(import sys)
(import os)
(import json)
(import redis)
(from urllib.parse import urlparse)

(mac errsafe (expr)
  `(on-err (fn (c) nil)
           (fn () ,expr)))

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

(def make-redis (url)
  (let (url (or url (get os.environ "REDIS_URL"))
        o (urlparse url))
    (redis.Redis host: o.hostname port: o.port password: o.password)))

(defvar r* (make-redis))

(def load-item (id)
  (let-when x (r*.get (cat id "_faces"))
    (with r (json.loads x)
      (set (get r 'id) id))))

(defvar items* (obj))

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
       (while (>= (dec ,n) 0)
         (let-when ,i (item ,n)
           ,@body)))))

(import hashlib)

(def shash (s)
  (let (s (if (string? s)
              (s.encode "utf-8")
             s)
        h (hashlib.md5 s))
    (h.hexdigest)))
  
(defvar encoder-cache-dir* (if (os.path.isdir "/drive") "/drive/stylegan-server-cache" "/tmp/cache"))
(defvar encoder-open-url-verbose* false)

(os.makedirs encoder-cache-dir* exist_ok: true)

(import dnnlib)

(def open-url (url)
  (dnnlib.util.open_url url verbose: encoder-open-url-verbose* cache_dir: encoder-cache-dir*))

(import requests)

(def urlbytes (url)
  (let-when r (requests.get url)
    (when (= r.status-code 200)
      (to-bytes r.content))))

;(def urlbytes (url)
; (with (open-url url) as f
;   (to-bytes (f.read))))

(def hash-url (url)
  (let bs (urlbytes url)
    (if (is? bs) (shash bs) (error (cat "Couldn't fetch url " url)))))

(def todisk (fname value)
  (if (nil? value)
      (errsafe (os.remove fname))
    (do (os.makedirs (os.path.dirname fname) exist_ok: true)
        (with (open (cat fname ".tmp") "wb") as f
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
  (let-when bs (urlbytes url)
    (PIL.Image.open (BytesIO bs))))

(import scrap)

(from glob import glob)

(import align_images_ext)

(def align-image (url)
  (with-temp-dir aligned-src-dir
    (with-temp-dir aligned-dst-dir
      (let-when img (url-image url)
        (set img (img.convert "RGB"))
        (img.save (p aligned-src-dir "image.jpg") "JPEG")
        (scrap.system "python3" (list "align_images.py" aligned-src-dir aligned-dst-dir))
        (or (let-when aligned-fname (hd (glob (p aligned-dst-dir "*")))
              (PIL.Image.open aligned-fname))
            (align_images_ext.square_thumb img 1024))))))

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
            (|np.load| latent-fname)))))))

(def latent-image (url)
  (let-when bs (get-url-key url 'latent)
    (return (np.array (obj-from-bytes bs))))
  (let-when img (aligned-image url)
    (let-when latent (make-latent-image img)
      (let-when bs (obj-to-bytes latent)
        (put-url-key url 'latent bs))
      latent)))

(def read-image (l)
  (if (hd? l 'image)
      (let url (at l 1)
        (let latent (latent-image url)
          (if latent
              (list '|np.array| (list 'quote (latent.tolist)))
            (list '|np.array| ()))))
    l))

(import reader)

(def read-safe (s)
  (if (string? s)
      (either (reader.maybe-number s) s)
    s))

(from importlib import reload)

(from IPython.display import display)

(def gen-image (latent)
  (let-when img-result (scrap.system "python3" (list "generator.py") stdin: (to-bytes (obj-to-bytes latent)))
    (let-when img (image-from-bytes img-result)
      img)))

(def urlpost (url data)
  (requests.post url: url data: (to-bytes (or data ""))))

(def gen-image (latent)
  (import generator)
  (generator.generate-image latent))

(def process-image (latent id)
  (print (cat id ": Generating image from " (repr latent)))
  (let-when img (gen-image latent)
    (print (cat id ": Generated image"))
    (print img)
    (r*.set (cat id "_faces:result") (image-to-bytes img))
    (let-when cb (item id 'callback)
      (print (cat "Pinging " cb))
      (urlpost cb 'ok))
    (display (img.resize (list 256 256)))
    'ok))

(def eval-image (url)
  (print (cat "Evaluating image " url))
  ;(np.array ()))
  (latent-image url))

(mac image (url)
  `(eval-image ',url))

(def item-op (i)
  `(process-image ,(maptree idfn read-safe (item i 'op)) ',(item i 'id)))

(def read-op (i)
  (item-op i))

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
  (let (id (item i 'id)
        k (cat id "_faces:result"))
    (unless (r*.keys k)
      (let-when op (read-op i)
        (print (cat "Processing " k ": " (str (expand op))))
        (set (get ops* id) op)
        (set (get results* id) (eval op))))))

(def process-on-error (i c)
  (let id (item i 'id)
    (|print| (cat id " ERROR:" (repr c)))
    (unless (r*.get (cat id "_faces:result"))
      (let-when img (PIL.Image.new size: '(16 16) mode: "RGB" color: "black")
        (r*.set (cat id "_faces:result") (image-to-bytes img))))
    (let-when cb (item i 'callback)
      (print (cat "Pinging " cb))
      (urlpost cb 'error))))

(def process-items ()
  (os.system "git pull")
  (load "encoder.l")
  (for-each-item i
    (on-err (fn (c) (process-on-error i c))
            (fn () (process-item i)))))
