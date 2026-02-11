import http.server
import socketserver

PORT = 8090

class Handler(http.server.SimpleHTTPRequestHandler):
    pass

with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()

    