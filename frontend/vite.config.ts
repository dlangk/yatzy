import { defineConfig, type Plugin } from 'vite'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..')

const MIME: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
}

/** Serve a directory at a URL prefix. */
function serveDir(prefix: string, dir: string): Plugin {
  return {
    name: `serve-${prefix.replace(/\//g, '-')}`,
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(prefix)) return next()

        let relPath = req.url.slice(prefix.length) || 'index.html'
        // Strip query string
        relPath = relPath.split('?')[0]
        if (relPath === '' || relPath.endsWith('/')) relPath += 'index.html'

        const filePath = path.join(dir, relPath)

        // Security: prevent path traversal
        if (!filePath.startsWith(dir)) return next()

        if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
          // Try with index.html for directory paths
          const indexPath = path.join(filePath, 'index.html')
          if (fs.existsSync(indexPath)) {
            const content = fs.readFileSync(indexPath)
            res.setHeader('Content-Type', 'text/html')
            res.end(content)
            return
          }
          return next()
        }

        const ext = path.extname(filePath)
        const mime = MIME[ext] || 'application/octet-stream'
        const content = fs.readFileSync(filePath)
        res.setHeader('Content-Type', mime)
        res.end(content)
      })
    },
  }
}

export default defineConfig({
  base: '/yatzy/play/',
  server: {
    port: 5173,
    proxy: {
      '/yatzy/api': {
        target: 'http://localhost:9000',
        rewrite: (p) => p.replace(/^\/yatzy\/api/, ''),
      },
    },
  },
  plugins: [
    serveDir('/yatzy/profile/', path.join(root, 'profiler')),
    serveDir('/yatzy/', path.join(root, 'treatise')),
  ],
})
