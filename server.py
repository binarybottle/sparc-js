#!/usr/bin/env python3
# coding: utf-8

import http.server
import socketserver

PORT = 8000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()
        
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

print(f"Starting server at http://localhost:{PORT}")
print("Press Ctrl+C to quit")

# Create server
handler = CORSHTTPRequestHandler
with socketserver.TCPServer(("", PORT), handler) as httpd:
    httpd.serve_forever()