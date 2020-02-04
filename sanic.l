(import lumen)
(import json as JSON)
(from sanic import Sanic)
(from sanic.response import json)

(defconst app (Sanic name: "pymen"))

(@ (app.route "/foo"))
(async def test (request)
  (json (obj hello: "world")))

(@ (app.route "/eval"))
(async def server-eval (request)
  ;(let ((ok v) (guard (eval (read-string request.args))))
    ;(if ok
  (json (obj result: (eval (lumen.reader.read-string (hd (get request.args 'code)))))))

;if __name__ == "__main__":
;(app.run host: "0.0.0.0" port: 8000)
