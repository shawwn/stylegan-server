(%block class Namespace (do))

(defvar args (Namespace))

(set args.batch_size 1 args.randomize_noise false args.tile_dlatents false args.clipping_threshold 2.0 args.model_res 1024 args.lr 0.02 args.decay_rate 0.9 args.decay_steps 10 args.image_size 256 args.use_vgg_layer 9 args.use_vgg_loss 0.4 args.face_mask false args.use_grabcut true args.scale_mask 1.5 args.mask_dir "masks" args.use_pixel_loss 1.5 args.use_mssim_loss 100 args.use_lpips_loss 100 args.use_l1_penalty 1)

(import os)
(import pickle)
(import PIL.Image)
(import numpy as np)
(import tensorflow as tf)
(import dnnlib)
(import dnnlib.tflib as tflib)
(import config)
(from encoder.generator_model import Generator)
(from encoder.perceptual_model import PerceptualModel load_images)
(from keras.models import load_model)

;(import matplotlib.pyplot as plt)

(defconst URL_FFHQ "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ")
(defconst URL_PERC "https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2")

(def load-generator ()
  (with (dnnlib.util.open_url URL_FFHQ cache_dir: config.cache_dir) as f
    (pickle (.load f))))

(def load-perceptual ()
  (with (dnnlib.util.open_url URL_PERC cache_dir: config.cache_dir) as f
    (pickle (.load f))))

(def init-generator ()
  (global generator_network)
  (global discriminator_network)
  (global Gs_network)
  (global generator)
  (set generator_network, discriminator_network, Gs_network (load-generator))
  (set generator (Generator Gs_network batch_size: 1 randomize_noise: false))
  generator)

(defvar reference-images*
  (list "/drive/stylegan-server-cache/217677dfe303180f7736ca5ecf0868a3_aligned"))

(def init-perceptual ()
  (global perc_model)
  (global perceptual_model)
  (set perc_model (load-perceptual))
  (set perceptual_model (PerceptualModel args perc_model: perc_model batch_size: args.batch_size))
  (perceptual_model.build_perceptual_model generator)
  (perceptual_model.set_reference_images reference-images*)
  )

(defvar ff-model* nil)
(defvar resnet-path* "data/finetuned_resnet.h5")
(defconst resnet-download-url* "https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb")

(def init-resnet ()
  (global ff-model*)
  (unless (os.path.exists resnet-path*)
    (os.makedirs (os.path.dirname resnet-path*) exist_ok: true)
    (with (dnnlib.util.open_url resnet-download-url* cache_dir: "cache") as f
      (with (open resnet-path* "wb") as dst
        (dst.write (f.read)))))
  (unless ff-model*
    (set ff-model* (load_model resnet-path*)))
  ff-model*)

(def image? (x)
  (isinstance x PIL.Image.Image))

(def file? (x)
  (os.path.isfile x))

(def numpy? (x)
  (isinstance x np.ndarray))

(def fetch-image (url)
  (if (image? url) url
      (file? url) (PIL.Image.open url)
    (let resp (GET url)
      (when (= resp.status_code 200)
        (PIL.Image.open (BytesIO resp.content))))))

(def convert-image (img)
  (img.convert "RGB"))

(def image-to-numpy (img)
  (if (numpy? img) img (np.array (fetch-image img))))

(def expand-image (x)
  (let-when img (fetch-image x)
    (let img (convert-image img)
      (let-when data (image-to-numpy img)
        (if (= (# data.shape) 3)
            (np.expand_dims data axis: 0)
          data)))))

(from keras.applications.resnet50 import preprocess_input)

(defconst default-dlatent* (np.zeros shape: '(1 18 512)))

(def estimate-dlatent (img)
  (let result (let-when model (init-resnet)
                (let-when data (expand-image img)
                  (model.predict (preprocess_input data))))
    (if (is? result) result (default-dlatent*.copy))))

;(tflib.init_tf)
;(with (dnnlib.util.open_url URL_FFHQ, cache_dir=config.cache_dir) as f
;  (set generator_network, discriminator_network, Gs_network (pickle.load f)))

(defvar latents* (Namespace))

(set latents*.trump (np (.load "ffhq_dataset/latent_representations/donald_trump_01.npy"))
     latents*.hillary (np (.load "ffhq_dataset/latent_representations/hillary_clinton_01.npy")))

(step x (list "smile" "gender" "age")
  (let path (cat "ffhq_dataset/latent_directions/" x ".npy")
    (setattr latents* x (np (.load path)))))

(from glob import glob)
(for k, v in (list (list (hd ((at (x (.split "/")) 1) (.split "_direction"))) (np (.load x))) for x in (glob "trained_directions/*.npy"))
  (setattr latents* k v))

(def file-to-bytes (path)
  (with (open path "rb") as f
    (f.read)))

(import json)

(def file-to-latent (path)
  (np.array (json.loads (file-to-bytes path))))

(def load-latents ()
  (let i 0
    (for path in (sorted (hd (list (glob "/drive/stylegan-server-cache/*_latent"))))
    ;(for k, v in (list (list (hd ((at (x (.split "/")) -1) (.split "_latent"))) (np (.load x))) for x in 
      (let (name (% "i%03d" (inc i))
            v (file-to-latent path))
        (setattr latents* name v)))))

(def generate-image (latent)
  (let (latent (np.array latent)
        latent (latent.reshape '(1 18 512)))
    (generator.set_dlatents latent)
    (let (img-array (at (generator.generate_images) 0)
          img (PIL.Image.fromarray img-array "RGB"))
      img)))

(defvar image-path* (os.path.join "/tmp" "images"))
(defvar error-image* (PIL.Image.new size: '(16 16) mode: "RGB" color: "black"))
(defvar saved-image* error-image*)

(def next-path (base)
  (with i 0
    (while (os.path.exists (% base i))
      (inc i))))

(def image-path (idx)
  (let (fmt (os.path.join image-path* "image_%05d.png")
        idx (if (is? idx) idx (next-path fmt)))
    (% fmt idx)))

(def save-image (img idx)
  (os.makedirs image-path* exist_ok: true)
  (let fname (image-path idx)
    (print (cat "Saving " fname))
    (img.save fname "PNG")
    (global saved-image*)
    (set saved-image* (img.resize '(256 256)))
    img))

(def calculate-latent (spec)
  (with result (np.zeros shape: '(1 18 512))
    (step (weight x) (pair spec)
      (let-when latent (if (string? x) (getattr latents* x)
                           (array? x) x)
        (inc result (* weight latent))))))

(def generate-image-from-spec (spec)
  (let (latent (calculate-latent spec)
        img (generate-image latent))
    ;(save-image img idx)))
    img))

(def randf ()
  (np.random.uniform))

(def generate-random-image ()
  (let spec (list (randf) 'trump (randf) 'hillary)
    (generate-image-from-spec spec)))

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
    (let fname (os.path.join tmp-image-dir "image")
      (apply img.save fname args)
      (with (open fname "rb") as f
        (f.read)))))
  
(import websockets)
(import datetime)
(import asyncio)

(defconst regen-delay* 3)

(async def handle-serve-1 (websocket path)
  (let now (cat (datetime.datetime (.utcnow) (.isoformat)) "Z")
    (await (websocket.send now))
    (await (asyncio.sleep (* (randf) regen-delay*)))))

(async def handle-serve-1 (websocket path)
  (save-image (generate-random-image) 0)
  (await (websocket.send (image-to-bytes saved-image* "PNG")))
  (await (asyncio.sleep (* (randf) regen-delay*))))

(defconst ellipsize-limit* 240)

(def ellipsize (s limit)
  (let n (either limit ellipsize-limit*)
    (if (> (# s) n)
        (cat (clip s 0 n) "...")
      s)))

(async def handle-serve-1 (websocket path)
  (let x (await (websocket.recv))
    (print (ellipsize (repr x)))
    (save-image (generate-random-image) 0)
    (await (websocket.send (image-to-bytes saved-image* "PNG")))))

(import reader)

(def image? (x)
  (isinstance x PIL.Image.Image))

(def bytes? (x)
  (isinstance x bytes))

(import inspect)

(define-global awaitable? (x)
  (inspect.isawaitable x))

(async def awaitable (x)
  (if (awaitable? x) (await x) x))

(def gathered (x)
  (if (awaitable? x) x ((async fn () (if (function? x) (x) x)))))

(def current-task ()
  (asyncio.Task.current-task))

(defvar id-count* 1)

(def get-id (x)
  (global id-count*)
  (if (hasattr x "lumen_id")
      (getattr x "lumen_id")
      (with i (inc id-count*)
        (setattr x "lumen_id" i))))

(def current-task-id ()
  (let task (current-task)
    (if task (get-id task) 1)))

(%block class (Tagged Namespace)
  (def __init__ (self tag rep)
    (set self.tag tag
         self.rep rep)
    nil)
  (def __repr__ (self)
    (cat "Tagged(" (repr self.tag) ")")))

(def tagged? (x)
  (and (hasattr x 'tag) (hasattr x 'rep)))

(def kind (x)
  (if (tagged? x) x.tag (type x)))

(def tag (x y)
  (if (= (kind x) y) x (Tagged y x)))

(def rep (x)
  (if (tagged? x) x.rep x))

(def make-thread-cell-value (cell value)
  (with self (tag nil 'thread-cell-value)
    (set self.cell cell
         self.value value
         self.rep self)))

(def thread-cell-value? (v)
  (= (kind v) 'thread-cell-value))

(import weakref)

(defvar preserved-thread-cells* (list))
(defvar preserved-thread-cell-values* (obj))

(def add-preserved-thread-cell (cell)
  (add preserved-thread-cells* (weakref.ref cell)))

(def set-preserved-thread-cell-values (id vals)
  (set (get preserved-thread-cell-values* id) vals))

(def grab-preserved-thread-cell-values ()
  (list (make-thread-cell-value cell (thread-cell-ref cell))
        for cell in (array (map call preserved-thread-cells*))))

(def get-preserved-thread-cell-values (id)
  (has preserved-thread-cell-values* id))

(def find-preserved-thread-cell-value (vals cell)
  (step x vals
    (when (= x.cell cell)
      (return x.value)))
  cell.default)

(def current-preserved-thread-cell-values args
  (if (none? args)
      (grab-preserved-thread-cell-values)
      (let vals (hd args)
        (step x preserved-thread-cells*
          (let-when x (x)
            (let v (find-preserved-thread-cell-value vals x)
              (thread-cell-set x v)))))))

(def make-thread-cell (v preserved?)
  (with self (tag nil 'thread-cell)
    (set self.default v
         self.values (obj)
         self.preserved preserved?
         self.rep self)
    (when preserved?
      (add-preserved-thread-cell self))))

(def thread-cell? (v)
  (= (kind v) 'thread-cell))

(def thread-cell-ref (v id)
  (let id (if (is? id) id (current-task-id))
    (if (has? v.values id)
        (get v.values id)
        ;v.preserved
        ;(step x (get-preserved-thread-cell-values id)
        ;  (when (= x.cell v)
        ;    (return x.value)))
      v.default)))

(def thread-cell-set (cell v)
  (unless (thread-cell? cell)
    (error "Expected thread-cell"))
  (let id (current-task-id)
    (if (nil? v)
        (wipe (get cell.values id))
      (set (get cell.values id) v))))

(def create-task (f)
  (let (loop (event-loop)
        vals (current-preserved-thread-cell-values)
        thunk ((async fn ()
                 (current-preserved-thread-cell-values vals)
                 (await (gathered f))))
        task (loop.create-task thunk))
    task))

(def schedule (f)
  (create-task f))

(define-macro thread body
  `(schedule ((async fn () ,@body))))

(defvar websocket* (make-thread-cell nil true))

(def current-socket args
  (if (none? args)
      (thread-cell-ref websocket*)
    (thread-cell-set websocket* (hd args))))

(async def send-image (img)
  (let val (image-to-bytes (img.resize '(256 256)) "PNG")
    (await ((current-socket) (.send val)))
    img))

(defvar data* (list))

(async def repl-print (form)
  (let ((ok v ex) (guard ((idx compiler eval) `((async fn () ,form)))))
    (if (not ok)
        (print-exception v ex)
        (is? v)
        (let v (await v)
          (do (print (ellipsize (repr v))) v)))))

(async def handle-serve-1 (websocket path)
  (let x (await (websocket.recv))
    (if (bytes? x)
        (do (add data* x)
            (await (websocket.send (cat "data*." (edge data*)))))
      (let form (reader.read-string x)
        (print (ellipsize (repr form)))
        (let (result (await (repl-print form))
              result (if (image? result) result error-image*)
              ;result (result.resize '(256 256)))
              )
          ;(await (websocket.send (image-to-bytes result "PNG"))))))))
          (await (send-image result)))))))

(async def handle-serve (websocket path)
  (while true
    ;(load "repl.l")
    (current-socket websocket)
    (await (handle-serve-1 websocket path))))

(defvar start-server* (websockets.serve (fn args (apply handle-serve args)) "0.0.0.0" 5679 max_queue: nil read_limit: (* 100 1024 1024) write_limit: (* 100 1024 1024) max_size: (* 100 1024 1024)))

(defvar server* nil)

(def event-loop ()
  (asyncio (.get-event-loop)))

(def serve ()
  (global server*)
  (set server* (or server* (asyncio (.get-event-loop) (.run-until-complete start-server*)))))

(def setup ()
  (global sess)
  (tflib.init_tf)
  (set sess (tf.get_default_session))
  (init-generator)
  (init-perceptual)
  (init-resnet)
  (serve))

(def run-forever ()
  (asyncio (.get-event-loop) (.run-forever)))

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

(def get-landmarks (img)
  (list marks for marks in (landmarks-detector*.get-landmarks img)))

(def face-align (img)
  (let marks (get-landmarks img)
    (if (none? marks)
        (do (print "No face detected")
            img)
      (image-align img (hd marks)))))


